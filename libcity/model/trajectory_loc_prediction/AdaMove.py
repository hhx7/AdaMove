import heapq
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from libcity.model.abstract_model import AbstractModel
from logging import getLogger


class Attn(nn.Module):
    """
    Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
    """

    def __init__(self, method, hidden_size, device):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.device = device
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history, value=None, history_len=None):
        """[summary]

        Args:
            out_state (tensor): batch_size * state_len * hidden_size
            history (tensor): batch_size * history_len * hidden_size

        Returns:
            [tensor]: (batch_size, state_len, history_len)
        """
        if self.method == 'dot':
            history = history.permute(0, 2, 1)  # batch_size * hidden_size * history_len
            attn_energies = torch.bmm(out_state, history)
        elif self.method == 'abs':
            distance = 1/(torch.abs(out_state - history)+1)
            attn_energies = distance
        elif self.method == 'general':
            history = self.attn(history)
            history = history.permute(0, 2, 1)
            attn_energies = torch.bmm(out_state, history)
        
        if history_len is None:
            # attn_energies = F.relu(attn_energies)
            attn_energies = F.softmax(attn_energies, dim=-1)
            return attn_energies
        else:
            attn_weights_softmax = torch.zeros_like(attn_energies)
            for i in range(len(history_len)):
                attn_weights_softmax[i, :, :history_len[i]] = F.softmax(attn_energies[i, :, :history_len[i]], dim=-1) 
            return attn_weights_softmax.bmm(value)

@torch.jit.script
def softmax_entropy_batch(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    log_probs = torch.log_softmax(x, dim=2)
    return -(torch.exp(log_probs) * log_probs).sum(2)


class TTA:
    def __init__(self, fc_final, loc_size, device, filter_M=6) -> None:
        self.fc_final = fc_final
        self.num_classes = loc_size
        self.device = device
        self.filter_M = filter_M

    def filter_samples_with_heap(self, ent, y_hat, filter_K, strategy, device):
        """
        按类别维护独立的小堆，提升筛选效率，并确保稳定排序。
        """
        # 策略确定排序方向
        multiplier = -1 if strategy == 'sim' else 1
        
        # 按类别存储堆
        class_heap_dict = {}
        for i, e in enumerate(ent):
            class_idx = y_hat[i].item()
            if class_idx not in class_heap_dict:
                class_heap_dict[class_idx] = []
            # 保证堆中的每个元素为 (熵值, 索引)，通过索引保证稳定排序
            heapq.heappush(class_heap_dict[class_idx], (multiplier * e, i))
        
        # 按类别筛选最多 filter_K 个样本
        selected_indices = []
        for heap in class_heap_dict.values():
            # 逐个弹出堆顶元素，保持原始索引顺序
            selected_indices.extend([heapq.heappop(heap)[1] for _ in range(min(filter_K, len(heap)))])
        
        return selected_indices

    def select_supports_q(self, supports, y_hat, ent, strategy='sim'):


        selected_indices = self.filter_samples_with_heap(ent, y_hat, self.filter_M, strategy, self.device)
        
        # 将选择的索引转换为张量并移至指定设备
        selected_indices = torch.tensor(selected_indices).to(self.device)
        
        # 返回选择的 supports 和 labels
        return supports[selected_indices], y_hat[selected_indices]
    
    def forward(self, x, y, new_x, strategy='sim', loc_len=None):
        # update labels and supports
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        pos_pred_y = torch.zeros(batch_size, self.num_classes).to(self.device)

        if strategy == 'sim':
            # similarity
            ent_batch = torch.bmm(x, new_x.unsqueeze(2)).squeeze(2)
        else:
            # entroy
            p_batch = self.fc_final(x) 
            ent_batch = softmax_entropy_batch(p_batch)
            mask = torch.arange(seq_len).unsqueeze(0) >= loc_len.unsqueeze(1)  # [batch_size, seq_len]

            # 将填充的部分置为最大值
            max_value = torch.finfo(ent_batch.dtype).max  # 获取当前 dtype 的最大值
            ent_batch = ent_batch.masked_fill(mask, max_value)
                

        if y is None:
            # no labels
            out = self.fc_final(x).squeeze(1)
            score = F.log_softmax(out, dim=2)
            _, y = torch.topk(score, 1, dim=2)
            y = y.squeeze(2)


        x, y, new_x = x.detach(), y.detach(), new_x.detach()
        for b in range(batch_size):
            x_row = x[b]
            new_x_row = new_x[b].unsqueeze(0)
            ent = ent_batch[b]
            
            pos_supports, pos_y_hat = self.select_supports_q(x_row, y[b], ent, strategy)


            labels = torch.nn.functional.one_hot(pos_y_hat, num_classes=self.num_classes).float()
            weights = (pos_supports.t() @ (labels))
            # 计算每个类别的样本数
            class_counts = labels.sum(dim=0)
            weights = weights / (class_counts.unsqueeze(0) + 1)
            zero_mask = weights == 0 
            weights[zero_mask] = self.fc_final.weight.data.T[zero_mask]
            # cos
            new_x_row = new_x_row / new_x_row.norm(p=2)
            weights = weights / weights.norm(p=2)
            pos_pred_y[b] = new_x_row @ weights
        return pos_pred_y

class AdaMove(AbstractModel):
    def __init__(self, config, data_feature):
        super(AdaMove, self).__init__(config, data_feature)
        self.device = config['device']
        self.loc_size = data_feature['loc_size']
        self.loc_emb_size = config['loc_emb_size']
        self.tim_size = data_feature['tim_size']
        self.tim_emb_size = config['tim_emb_size']
        self.hidden_size = config['hidden_size']
        self.attn_type = config['attn_type']
        
        self.batch_size = config['batch_size']
        self.uid_size = data_feature['uid_size']
        self.uid_emb_size = config['uid_emb_size']
        self.loc_pad = data_feature['loc_pad']
        self.tim_pad = data_feature['tim_pad']
        self.test_time = config['test_time']
        self.filter_M = config['filter_M']
        self.lam = config['lambda']
        self.strategy = config['strategy']

        self.emb_loc = nn.Embedding(
            self.loc_size, self.loc_emb_size, 
            padding_idx=data_feature['loc_pad'], device=self.device)
        self.emb_tim = nn.Embedding(
            self.tim_size, self.tim_emb_size,
            padding_idx=data_feature['tim_pad'], device=self.device)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size, device=self.device)
        self.input_size = self.loc_emb_size + self.uid_emb_size + self.tim_emb_size
        
        self.rnn_encoder = nn.LSTM(self.input_size, self.hidden_size, 1, device=self.device)

        
        self.attn = Attn(self.attn_type, self.hidden_size, self.device)
        self.fc_final = nn.Linear(self.hidden_size + self.uid_emb_size, self.loc_size, bias=True, device=self.device)

         
        self.tta = TTA(self.fc_final, self.loc_size, self.device, self.filter_M)
 

        self._logger = getLogger()
        self.avg_time = []
        self.total_time = []


    def forward(self, batch): 
        loc = batch['current_loc']  
        tim = batch['current_tim']
        uid = batch['uid']
        history_loc = batch['history_loc']
        history_tim = batch['history_tim']
        history_len = batch.get_origin_len('history_loc')
        loc_len = batch.get_origin_len('current_loc')
        batch_size = loc.shape[0]
        target = batch['target']

        x_history_loc = batch['x_history_loc']
        x_history_tim = batch['x_history_tim']
        x_history_len = batch.get_origin_len('x_history_loc')
        

        if not self.training and self.test_time:
            # cat prefix
            total_len_tmp = x_history_len.copy()
            for i in range(batch_size):
                total_len_tmp[i] = x_history_len[i] + loc_len[i]

            max_seq_len = max(total_len_tmp)
            total_loc_tmp = torch.zeros(batch_size, max_seq_len).to(self.device)
            total_tim_tmp = torch.zeros(batch_size, max_seq_len).to(self.device)
            total_loc_tmp.fill_(self.loc_pad)
            total_tim_tmp.fill_(self.tim_pad)
            for i in range(batch_size):
                total_loc_tmp[i,:total_len_tmp[i]] = torch.cat((x_history_loc[i,:x_history_len[i]].unsqueeze(0), loc[i,:loc_len[i]].unsqueeze(0)), dim=1)
                total_tim_tmp[i,:total_len_tmp[i]] = torch.cat((x_history_tim[i,:x_history_len[i]].unsqueeze(0), tim[i,:loc_len[i]].unsqueeze(0)), dim=1)

            loc = total_loc_tmp.long()
            tim = total_tim_tmp.long()
            loc_len = total_len_tmp
        loc_len = torch.tensor(loc_len)

        start_time = time.time()

        h1 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        h2 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        c1 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        c2 = torch.zeros(1, batch_size, self.hidden_size, device=self.device)


        uid_emb = self.emb_uid(uid)
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)

        x_uid_emb = uid_emb.unsqueeze(1).expand(loc_emb.shape[0], loc_emb.shape[1], -1)
        
        x = torch.cat((loc_emb, tim_emb, x_uid_emb), 2).permute(1, 0, 2)
        pack_x = pack_padded_sequence(x, lengths=loc_len, enforce_sorted=False)
        hidden_states, (h1, c1) = self.rnn_encoder(pack_x, (h1, c1))
        hidden_states, hidden_state_len = pad_packed_sequence(
            hidden_states, batch_first=True)
        
        final_out_index = loc_len - 1
        final_out_index = final_out_index.reshape(final_out_index.shape[0], 1, -1)
        final_out_index = final_out_index.repeat(1, 1, self.hidden_size).to(self.device)
        final_hidden_state = torch.gather(hidden_states, 1, final_out_index).squeeze(1)
        
        if not self.test_time:
            history_loc_emb = self.emb_loc(history_loc)
            history_tim_emb = self.emb_tim(history_tim)
            history_uid_emb = uid_emb.unsqueeze(1).expand(history_loc_emb.shape[0], history_loc_emb.shape[1], -1)

            history_x =  torch.cat(
                (history_loc_emb, history_tim_emb, history_uid_emb), 2).permute(1, 0, 2)
            history_pack_x = pack_padded_sequence(history_x, lengths=history_len, enforce_sorted=False)
            history_hidden_states, (h2, c2) = self.rnn_encoder(history_pack_x, (h2, c2))
            history_hidden_states, hidden_state_len = pad_packed_sequence(
                history_hidden_states, batch_first=True)
            
            mask = (target.unsqueeze(1) != loc).unsqueeze(-1).expand(loc.shape[0], loc.shape[1], hidden_states.shape[-1])
            hidden_states = hidden_states * mask
            # batch_size * state_len * history_len
            attn_weights = self.attn(hidden_states, history_hidden_states)
            # batch_size * state_len * hidden_size
            context = attn_weights.bmm(history_hidden_states)

            
            
    
            # # contrast loss
            # # InfoNCE
            pos_state = context[torch.arange(context.size(0)), loc_len - 1, :].clone()
            context[torch.arange(context.size(0)), loc_len - 1, :] = 0
            pos_sim = F.cosine_similarity(final_hidden_state, pos_state, dim=-1)
            neg_sim = F.cosine_similarity(final_hidden_state.unsqueeze(1), context, dim=-1)
        
            contrast_loss = (torch.logsumexp(neg_sim, dim=1) - pos_sim).mean()

        new_x = torch.cat((final_hidden_state, uid_emb), dim=1)
 
        if not self.training and self.test_time:
            final_uid_emb = uid_emb.unsqueeze(1).expand(batch_size, hidden_states.shape[1], -1)
            hidden_states = torch.cat((hidden_states, final_uid_emb), dim=-1)

            x = hidden_states[:, :-1]
            ground_labels = loc[:, 1:]
            if self.strategy == 'sim': 
                # +label +sim = PTTA
                y  = self.tta.forward(x, ground_labels, new_x, strategy=self.strategy)
                # -label +sim = w/ pesudo-label
                # y  = self.ttanuq.forward(x, None, new_x, strategy=self.strategy)
            else:
                # -label -sim == w/ T3A
                # y = self.ttanuq.forward(x, None, new_x, strategy=self.strategy, loc_len=loc_len)
                
                # +label -sim = w/ ent
                y  = self.tta.forward(x, ground_labels, new_x, strategy=self.strategy, loc_len=loc_len)
            score = F.log_softmax(y, dim=1)
            end_time = time.time()  
            batch_time = end_time - start_time

            self.avg_time.append(batch_time/batch_size)
            self.total_time.append(batch_time)
            if len(self.total_time) % 10 == 0:
                total_tim = sum(self.total_time)
                average_tim =  sum(self.avg_time)/ (len(self.avg_time))
                self._logger.info("Single Time = %.4f seconds; Average Time = %.4f seconds; Total Time = %.4f seconds", batch_time, average_tim, total_tim)
            return score, 0

        # LightMob
        y = self.fc_final(new_x)  # batch_size * loc_size
        score = F.log_softmax(y, dim=1)
        return score, self.lam * contrast_loss

    
    def predict(self, batch):
        score, _  = self.forward(batch)
        return score

    def calculate_loss(self, batch):
        criterion = nn.NLLLoss()
        scores, KLD  = self.forward(batch)
        l_con = criterion(scores, batch['target'])
        return l_con + KLD

    
