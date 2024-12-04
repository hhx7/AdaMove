# AdaMove
 
## Installation

Please execute the following command to get the source code.

```shell
git clone https://github.com/hhx7/AdaMove.git
cd AdaMove
```

Create environment using conda:
```shell
conda env create -f environment.yml
conda activate AdaMove
```
Or pip:
```shell
pip install -r requirements.txt
```

## Folder structure

We implement our model using [LibCity](https://github.com/LibCity/Bigscity-LibCity.git). The respective code files are organized into separate modules as follows:

- **Raw Data:**  
  Raw Foursquare datasets, including `NYC` and `TKY`, are stored in the `/raw_data/` folder.

- **Model Implementation:**  
  The implementation of our model `AdaMove` is located in `/libcity/model/trajectory_loc_prediction/AdaMove.py`.

- **Hyperparameter Configuration:**  
  Hyperparameter settings for `AdaMove` are saved in the JSON file located at `/libcity/config/model/traj_loc_pred/AdaMove_*.json`.


The main starting point for training and testing a model is the script `run_model.py`. This script facilitates the process of starting deep learning model training and evaluation within the [LibCity](https://github.com/LibCity/Bigscity-LibCity.git) framework.


## Running

The `run_model.py` script is used for training and evaluating a single model in LibCity. When running `run_model.py`, we need to specify the following three parameters: **task**, **dataset**, and **model**.

### Training and Evaluating on NYC
 
1. **Training:**

    ```sh
    python run_model.py --task traj_loc_pred --model AdaMove --dataset foursquare_nyc --config_file AdaMove_nyc
    ```

2. **Evaluating:**

    ```sh
    python run_model.py --task traj_loc_pred --model AdaMove --dataset foursquare_nyc --config_file AdaMove_nyc  --test_time true --gpu false --exp_id <exp_id>
    ```

    > **Note:** Replace `<exp_id>` with the trained model's experiment ID.

### Training and Evaluating on TKY

1. **Training:**

    ```sh
    python run_model.py --task traj_loc_pred --model AdaMove --dataset foursquare_tky --config_file AdaMove_tky
    ```

2. **Evaluating:**

    ```sh
    python run_model.py --task traj_loc_pred --model AdaMove --dataset foursquare_tky --config_file AdaMove_tky  --test_time true --gpu false --exp_id <exp_id>
    ```

    > **Note:** Similarly, replace `<exp_id>` with the trained model's experiment ID.

By using the above commands, you can train and evaluate the `AdaMove` model on specified datasets for trajectory location prediction tasks on NYC and TKY. Make sure to fill in the experiment ID (`exp_id`) correctly based on your actual training scenario to load the corresponding model for testing.
## The hyperparameters of AdaMove

The hyperparameters for AdaMove are defined in `/libcity/config/model/traj_loc_pred/AdaMove_*.json`.  

- Model Training  
  - `gpu`: Indicates whether to train the model on a GPU. Possible values: `true` or `false`.  
  - `gpu_id`: Specifies the GPU device to use.  
  - `learning_rate`: Sets the initial learning rate for the model.  
  - `lr_step`: Determines the schedule for adjusting the learning rate.  
  - `max_epoch`: Specifies the maximum number of training epochs.  
  - `optimizer`: Defines the optimizer used during training.  
  - `batch_size`: Sets the batch size for training.  

- Dataset Parameters  
  - `cache_dataset`: If `false`, the datasets will be processed from scratch.  
  - `x_history_fixed_length`: Specifies the fixed length of sessions during the testing stage.  
  - `window_size`: Defines the time interval for each session.  

- Training Hyperparameters  
  - `lambda`: Sets the weight of the contrastive loss.  
  - `loc_emb_size`: Specifies the embedding size for locations.  
  - `uid_emb_size`: Specifies the embedding size for user IDs.  
  - `tim_emb_size`: Specifies the embedding size for time.  
  - `hidden_size`: Sets the size of the encoder's hidden state.  

- Test-Time Hyperparameters  
  - `exp_id`: Identifies the trained model to use.  
  - `test_time`: Determines whether to evaluate the model using the `PTTA` module.  
  - `strategy`: Specifies the pattern matching strategy (`sim` or `ent`) when `test_time` is `true`.  
  - `filter_M`: Defines the maximum number of stored hidden states when `test_time` is `true`.  

