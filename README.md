# Active Feature Acquisition with Generative Surrogate Models
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

#### Nested Cross-Validation

To ensure that training the surrogate model runs in a reasonble (2-3 hours) amount of time, I decided to use nested cross validation to determine the top 20 cheap features for testing, training and evaluation. Here, the model I used was an MLP with 3 hidden layers and 300 hidden units, and the hyperparameters are the binary masks used to select the features. In nested cross-validation, I used a 3 fold inner cross-validation to select the best binary mask, and a 10 fold outer cross-validation for evaluation.

#### Chemistry Dataset with Top 20 Features (Classification)

- `molecule_20`: <br /> Dataset with top 20 features determined using nested CV.

**Top 20 Features Determined Using Nested CV:**
```
[234, 244, 322, 356, 393, 698, 725, 790, 792, 841, 80, 350, 465, 573, 583, 879, 901, 675, 147, 833]
```

#### Chemistry Dataset with Cheap/Expensive Features (Regression)

**Note**: All expensive features are normalized to range [-1,1]

1. `solvent_20_cheap`[^1]: <br /> Dataset with top 20 cheap features (including solvent descriptors) determined using nested CV.

2. `solvent_exp`[^1]: <br /> Dataset with all expensive features.

3. `solvent_20_HL`[^1]: <br /> Dataset with top 20 cheap features (including solvent descriptors) determined using nested CV <br /> and the expensive HOMO-LUMO `holu gap` feature. 
4. `solvent_20_exp`[^2]: <br /> Dataset with top 20 cheap features (including solvent descriptors) determined using nested CV <br /> and all the expensive features. 

**Top 20 Cheap Features Determined Using Nested CV:**
```
[1313, 352, 1808, 1594, 1724, 650, 824, 1476, 1379, 439, 45, 204, 584, 222_solv, 2, 1971, 249, 1754, 1357, 1573]
```

## Baselines

The `baselines` folder contains both the linear least squares and neural network models. <br /> We use these two models to set a baseline RMSE for training on the full chemistry cheap/expensive dataset (solvent). 

## Train and Test

You can train your own model by the scripts provided below.

### Cube

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
```

### Solvent (Regression)

Same as in `Cube` dataset except for these differences:

We will be using the ACflow Regression model and use the regression environment to train the agent using PPO.

- Train the ACflow model:
``` bash
python scripts/train_model.py --cfg_file=./exp/acflow/[dataset]/params.json
```

- Train the PPO Policy:
``` bash
python scripts/train_agent.py --cfg_file=./exp/ppo/[dataset]/params.json --env reg
```

**Important:** Ensure `--env reg` flag is selected.

[^1]: The acquisition cost of all features will be 0.
[^2]: The acquisition cost of cheap features will be 0 while the cost of the expensive features will be ≈ 18.125 (found using binary search).

### Results

The following files are in `results`.

- `evaluate`: <br /> There are two `.pkl` files, `test.pkl` and `train.pkl`, which are dictionaries that contain the rewards and state transitions over all episodes.
  - `transitions`: <br /> The `transitions` represent the accumulation of the masks over time steps of an episode. Thus, if a feature has a larger value in the `transitions` array, then it was selected earlier than a feature with a smaller value.
- `weights`: <br /> Folder that contains the weights for the MDP agent for reference.
- `params.json`: <br /> Configuration file to train the MDP agent.
- `learning_curve.png`: <br /> Graph of the rewards vs. episodes.
- `train.log`: <br /> Log file while training
