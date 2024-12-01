import os
from libcity.data.dataset.trajectory_encoder.abstract_trajectory_encoder import AbstractTrajectoryEncoder
from libcity.utils import parse_time

parameter_list = ['dataset', 'min_session_len', 'min_sessions', 'traj_encoder', 'cut_method',
                  'window_size', 'history_type', 'min_checkins', 'max_session_len']


class StandardTrajectoryEncoder(AbstractTrajectoryEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.uid = 0
        self.location2id = {}  # 因为原始数据集中的部分 loc id 不会被使用到因此这里需要重新编码一下
        self.id2location = {}
        self.user2id = {}
        self.id2user = {}
        self.loc_id = 0
        self.tim_max = 47  # 时间编码方式得改变
        self.history_type = self.config['history_type']
        self.x_history_type = self.config['x_history_type']  if 'x_history_type' in self.config else None
        self.uid_recode = self.config['uid_recode']
        self.loc_recode = self.config['loc_recode']
        self.history_fixed_length =  self.config['history_fixed_length'] if 'history_fixed_length' in self.config else None
        self.x_history_fixed_length =  self.config['x_history_fixed_length'] if 'x_history_fixed_length' in self.config else None
        self.feature_dict = {'history_loc': 'int', 'history_tim': 'int',
                             'current_loc': 'int', 'current_tim': 'int',
                             'target': 'int', 'target_tim': 'int', 'uid': 'int',
                             'x_history_loc': 'int', 'x_history_tim': 'int'}

        parameters_str = ''
        for key in parameter_list:
            if key in self.config:
                parameters_str += '_' + str(self.config[key])
        self.cache_file_name = os.path.join(
            './libcity/cache/dataset_cache/', 'trajectory_{}.json'.format(parameters_str))

        # 按距离构建图    
        self.dataset = self.config.get('dataset', '')
            


    
    def encode(self, uid, trajectories):
        """standard encoder use the same method as DeepMove

        Recode poi id. Encode timestamp with its hour.

        Args:
            uid ([type]): same as AbstractTrajectoryEncoder
            trajectories ([type]): same as AbstractTrajectoryEncoder
                trajectory1 = [
                    (location ID, timestamp, timezone_offset_in_minutes),
                    (location ID, timestamp, timezone_offset_in_minutes),
                    .....
                ]
        """
        # 直接对 uid 进行重编码
        if self.uid_recode:
            if uid not in self.user2id:
                self.user2id[uid] = self.uid
                self.id2user[self.uid] = uid
                self.uid += 1
            uid = self.user2id[uid]
        else:
            if uid not in self.user2id:
                self.user2id[uid] = uid
                self.id2user[uid] = uid
                self.uid += 1
            uid = int(uid)
        encoded_trajectories = []
        history_loc = []
        history_tim = []
        x_history_loc = []
        x_history_tim = []
        for index, traj in enumerate(trajectories):
            current_loc = []
            current_tim = []
            for point in traj:
                loc = point[4]
                now_time = parse_time(point[2])
                if self.loc_recode:
                    if loc not in self.location2id:
                        self.location2id[loc] = self.loc_id
                        self.id2location[self.loc_id] = loc
                        self.loc_id += 1
                else:
                    if loc not in self.location2id:
                        self.location2id[loc] = loc
                        self.id2location[loc] = loc
                        self.loc_id += 1
                current_loc.append(self.location2id[loc])
                time_code = self._time_encode(now_time)
                current_tim.append(time_code)
         
            # 完成当前轨迹的编码，下面进行输入的形成
            if index == 0:
                # 因为要历史轨迹特征，所以第一条轨迹是不能构成模型输入的
                if self.history_type == 'splice':
                    history_loc += current_loc
                    history_tim += current_tim

                if self.x_history_type == 'x_cut_off2splice':
                    x_history_loc.append(current_loc)
                    x_history_tim.append(current_tim)
                continue
            # 一条轨迹可以产生多条训练数据，根据第一个点预测第二个点，前两个点预测第三个点....
            for i in range(len(current_loc) - 1):
                trace = []
                target = current_loc[i+1]
 
                target_tim = current_tim[i+1]
                trace.append(history_loc.copy())
                trace.append(history_tim.copy())
                trace.append(current_loc[:i+1])
                trace.append(current_tim[:i+1]) 
                trace.append(target)
                trace.append(target_tim)
                trace.append(uid)
                
                x_history_loc_tmp = []
                x_history_tim_tmp = []

                if self.x_history_type == 'x_cut_off2splice':
                    if self.x_history_fixed_length is not None:
                        for t in x_history_loc[-self.x_history_fixed_length:]:
                            x_history_loc_tmp += t
                        for t in x_history_tim[-self.x_history_fixed_length:]:
                            x_history_tim_tmp += t
                

                trace.append(x_history_loc_tmp.copy())
                trace.append(x_history_tim_tmp.copy())

                encoded_trajectories.append(trace)

            if self.history_type == 'splice':
                history_loc += current_loc
                history_tim += current_tim
                

                
            if self.x_history_type == 'x_cut_off2splice':
                x_history_loc.append(current_loc)
                x_history_tim.append(current_tim)

            if self.history_fixed_length is not None:
                history_loc = history_loc[-self.history_fixed_length:]
                history_tim = history_tim[-self.history_fixed_length:]
                history_lon = history_lon[-self.history_fixed_length:]
                history_lat = history_lat[-self.history_fixed_length:]

        return encoded_trajectories

    def gen_data_feature(self):
        loc_pad = self.loc_id
        tim_pad = self.tim_max + 1
        if self.history_type == 'splice':
            self.pad_item = {
                'current_loc': loc_pad,
                'history_loc': loc_pad,
                'x_history_loc': loc_pad,
                'current_tim': tim_pad,
                'history_tim': tim_pad,
                'x_history_tim': tim_pad,
            }
        self.data_feature = {
            'loc_size': self.loc_id + 1,
            'tim_size': self.tim_max + 2,
            'uid_size': self.uid,
            'loc_pad': loc_pad,
            'tim_pad': tim_pad
        }

    def _time_encode(self, time):
        if 'time_encode' in self.config and self.config['time_encode'] == 'same':
            return time.hour
        else:
            if time.weekday() in [0, 1, 2, 3, 4]:
                return time.hour
            else:
                return time.hour + 24
