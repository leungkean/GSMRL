# Active Feature Acquisition with Generative Surrogate Models
## Get Started

### Prerequisites

The code requires `python3.7` to run. For the required python packages, refer to `requirements.txt`.

**IMPORTANT:** Ensure that you have `cudatoolkit=10.0` installed.
```
conda install -c anaconda cudatoolkit=10.0
```

### Download data and data preprocess

Download your training data into the data folder. You might need to convert the data file into a pickle file. The structure of the data should be a dictionary. The keys are 'train','valid', and 'test' and the values are the corresponding data tuple (x, y).
<br />
You might need to change the path for each dataset in `datasets` folder accordingly, in datasets folder, there is a corresponding file for each dataset that parse the data to fit the Tensorflow model.

#### Chemistry Dataset with Cheap/Expensive Features

- `solvent_20_cheap`: Dataset with top 20 cheap features (including solvent descriptors) determined using nested CV.

- `solvent_exp`: Dataset with all expensive features.

- `solvent_20_both`: Dataset with top 20 cheap features (including solvent descriptors) determined using nested CV and all expensive features.

**Top 20 Cheap Features Determined Using Nested CV:**
```
[1313, 352, 1808, 1594, 1724, 650, 824, 1476, 1379, 439, 45, 204, 584, 222_solv, 2, 1971, 249, 1754, 1357, 1573]
```

In datasets that include the HOMO-LUMO gap feature (`holu_gap`), I have explicitly included a `cost`. The acquisition cost of the `holu_gap` feature will be around 20X that of the cheap feature.

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

### Solvent

Same as in `Cube` dataset except for these differences

- Train the ACflow model

Change the directory for the corresponding dataset.

- Train the PPO Policy

**Important:** Add `--env reg` flag.

Change the directory for the corresponding dataset.
