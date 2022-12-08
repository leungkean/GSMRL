# Active Feature Acquisition with Generative Surrogate Models
Original: https://github.com/lupalab/GSMRL/tree/GSMRL
## Get Started

### Prerequisites

The code requires `python3.7` to run. For the required python packages, refer to `requirements.txt`.
```
pip install -r requirements.txt
```

**IMPORTANT:** Ensure that you have `cudatoolkit=10.0` installed.
```
conda install -c anaconda cudatoolkit=10.0
```

### Download data and data preprocess

Download your training data into the data folder. You need to convert the data file into a pickle file. The structure of the data should be a dictionary. The keys are 'train', 'valid', and 'test' and the values are the corresponding data tuple (x, y).
<br />
You need to change the path for each dataset in `datasets` folder accordingly, in datasets folder, there is a corresponding file for each dataset that parse the data to fit the Tensorflow model.

#### Psychology Dataset

The file [`clean_data.ipynb`](https://github.com/leungkean/GSMRL/blob/main/preprocess/clean_data.ipynb)
is used to properly combine the data and structure it as input for the GSMRL model.
Also the complete dataset is also given [here](https://github.com/leungkean/GSMRL/blob/main/preprocess/Daily%20Diary%20Long%20Form.csv).

## Generalizing GSMRL for Time Labeled Data

The psychology dataset (found in `data/psych_GSMRL.pkl`) is a dataset used to study the
short-term stability and fluctuation of personality pathology. 
The dataset contains collected 30 daily ratings of manifestations of personality pathology (DPDS)
rom 116 individuals over the course of 102 consecutive days.
The 30 DPDS ratings were then aggregated using multilayer factor analysis (MFA) and model fit statistics to create 9 daily domain scale.
Scores of each domain scale are used as the target variable y in the AFA regression problem.

Typically in AFA, we only observe and acquire features for a particular data instance.
However, an area of interest is to use the acquired features that predict target variables from a previous day
to help in the acquisition and prediction process for the current time.
As such, I have modified the code for this specific purpose.

The key idea is to first modify the dataset so that survey features and target variables over
a certain time interval or window for a given individual are combined into one large feature and target vector.
The time window is defined as the interval over `n` days, where `n` is manually entered by the user.
Using this modified dataset, I train the surrogate ACFlow model to learn the dependencies between
the combined features and target variables.
Afterwards, I train the RL with the surrogate model using PPO.
To mimic a machine detective, the RL agent is restricted to only being able to acquire features from a single day.
Only when the agent decides to make a prediction on the target variables for that particular day can it
move on to acquiring features for the next day.
Furthermore, the RL agent cannot go back in time and acquire more features.
The acquisition process stops when all target variables are predicted.
The aim of this setup is to enable the RL agent to use information from previous days to improve the
acquisition process and prediction of the current day in question.

### Implementation Details

1. The code used for the surrogate model in 
[`acflow_regressor.py`](https://github.com/leungkean/GSMRL/blob/main/models/acflow_regressor.py) 
was modified such that not only is the RMSE (root-mean squared error) for a single data instance given, 
but the RMSE for all days in a single data instance are also provided.
The RMSE for all days in the time window is given in [`rmse_list`](https://github.com/leungkean/GSMRL/blob/c8df40124162eec1d475d7d902d9df446f4dfa17/models/acflow_regressor.py#L59).
2. The main python file to train the RL agent, [`ppo.py`](https://github.com/leungkean/GSMRL/blob/main/agents/ppo.py)
is modified such that the function [`act`](https://github.com/leungkean/GSMRL/blob/c8df40124162eec1d475d7d902d9df446f4dfa17/agents/ppo.py#L56)
that determines which action to take only allows features from the current day to be acquired.
Only when the agent decides to take make a prediction for the current day does it allow the acquisition
of features for the next day. The acquisition process stops when all target variables are predicted.
Furthermore, I have included a small function to update the time that determines which time window
the agent is currently in called [`update_time`](https://github.com/leungkean/GSMRL/blob/c8df40124162eec1d475d7d902d9df446f4dfa17/agents/ppo.py#L106).
3. The regression environment found in [`reg_env.py`](https://github.com/leungkean/GSMRL/blob/main/envs/reg_env.py)
has been modified so that the [`step`](https://github.com/leungkean/GSMRL/blob/c8df40124162eec1d475d7d902d9df446f4dfa17/envs/reg_env.py#L116)
function allows for intermediate prediction and the prediction reward function [`_reg_reward`](https://github.com/leungkean/GSMRL/blob/c8df40124162eec1d475d7d902d9df446f4dfa17/envs/reg_env.py#L65)
has been modified to only give RMSE for the particular day.

<!---#### Nested Cross-Validation

To ensure that training the surrogate model runs in a reasonble (2-3 hours) amount of time, I decided to use nested cross validation to determine the top 20 cheap features for testing, training and evaluation. Here, the model I used was an MLP with 3 hidden layers and 300 hidden units, and the hyperparameters are the binary masks used to select the features. In nested cross-validation, I used a 3 fold inner cross-validation to select the best binary mask, and a 10 fold outer cross-validation for evaluation.--->

<!---#### Chemistry Dataset with Top 20 Features (Classification)

- `molecule_20`: <br /> Dataset with top 20 features determined using nested CV.

**Top 20 Features Determined Using Nested CV:**
```
[234, 244, 322, 356, 393, 698, 725, 790, 792, 841, 80, 350, 465, 573, 583, 879, 901, 675, 147, 833]
```

#### Chemistry Dataset with Cheap/Expensive Features (Regression)

**Note**: All expensive features are normalized to range [-1,1]

1. `solvent_20_cheap`[^1]: <br /> Dataset with top 20 cheap features (including solvent descriptors) determined using nested CV.

2. `solvent_exp`[^1]: <br /> Dataset with all expensive features.

3. `solvent_20_HL`[^2]: <br /> Dataset with top 20 cheap features (including solvent descriptors) determined using nested CV <br /> and the expensive HOMO-LUMO `holu gap` feature. 
4. `solvent_20_exp`[^2]: <br /> Dataset with top 20 cheap features (including solvent descriptors) determined using nested CV <br /> and all the expensive features. 

**Top 20 Cheap Features Determined Using Nested CV:**
```
[1313, 352, 1808, 1594, 1724, 650, 824, 1476, 1379, 439, 45, 204, 584, 222_solv, 2, 1971, 249, 1754, 1357, 1573]
```

## Baselines

The `baselines` folder contains both the linear least squares and neural network models. <br /> We use these two models to set a baseline RMSE for training on the full chemistry cheap/expensive dataset (solvent). --->

## Train and Test

You can train your own model by the scripts provided below.

<!---### Cube

- Train the ACflow model

``` bash
python scripts/train_model.py --cfg_file=./exp/acflow/cube/params.json
```

- Train the PPO Policy
``` bash
python scripts/train_agent.py --cfg_file=./exp/ppo/cube/params.json
```

### Molecule_20 (Classification)

- Train the ACflow model

``` bash
python scripts/train_model.py --cfg_file=./exp/acflow/molecule_20/params.json
```

- Train the PPO Policy
``` bash
python scripts/train_agent.py --cfg_file=./exp/ppo/molecule_20/params.json
```--->

### Regression

<!---Same as in `Cube` dataset except for these differences:--->

We will be using the ACflow Regression model and use the regression environment to train the agent using PPO.

- Train the ACflow model:
```
python scripts/train_model.py --cfg_file=./exp/acflow/[dataset]/params.json
```

- Train the PPO Policy:
``` bash
python scripts/train_agent.py --cfg_file=./exp/ppo/[dataset]/params.json --env reg
```

**Important:** Ensure `--env reg` flag is selected.

<!---### Variational GMM

We treat the data as a GMM and optimize the parameters to cluster the dataset into two mixture components:
one acquires only cheap features (z = 0) and the other component acquires both cheap and expensive features (z = 1).
In this case, we initialize the prior p(z) such that only cheap features are acquired around 10% of the time.--->

### Results

The following files are in `results`.

- `evaluate`: <br /> There are two `.pkl` files, `test.pkl` and `train.pkl`, which are dictionaries that contain the rewards and state transitions over all episodes.
  - `transitions`: <br /> The `transitions` represent the accumulation of the masks over time steps of an episode. Thus, if a feature has a larger value in the `transitions` array, then it was selected earlier than a feature with a smaller value.
- `weights`: <br /> Folder that contains the weights for the MDP agent for reference.
- `params.json`: <br /> Configuration file to train the MDP agent.
- `learning_curve.png`: <br /> Graph of the rewards vs. episodes.
- `train.log`: <br /> Log file while training

In the `evaluate` we also have a generated `question_trajectory.pkl` file.
This is a pickle file containing a dictionary that specifies the trajectory of questions
to ask before making a prediction.
The keys are the index of the particular data instance and the values are the tuples
of string that represent the questions to be asked.

<!---[^1]: The acquisition cost of all features will be 0.--->
<!---[^2]: The acquisition cost of all cheap features will be 0 and expensive features will be predetermined.--->
