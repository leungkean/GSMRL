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

- `solvent_cheap`: Dataset with all cheap features and all solvent descriptors.

- `solv_desc_cheap`: Dataset with solvent descriptors and the corresponding cheap features.

- `solvent_HL`: Dataset with all cheap features, `holu_gap`, and all solvent descriptors.

- `solv_desc_HL`: Dataset with solvent descriptors, corresponding cheap features, and `holu_gap`.

Cheap Features in `solv_desc`:
```
[80, 222, 294, 389, 650, 656, 674, 790, 807, 926, 1017, 1028, 1057, 1088, 1199, 1384, 1683, 1722, 1823, 1917, 1920]
```

In datasets that include the HOMO-LUMO gap feature (`holu_gap`), I have explicitly included a `cost`. The acquisition cost of the `holu_gap` feature will be around 20X that of the cheap feature.

## Train and Test

You can train your own model by the scripts provided below.

### Cube

- Train the ACflow model

``` bash
python scripts/train_model.py --cfg_file=./exp/cube/params.json
```

- Train the PPO Policy

``` bash
python scripts/train_agent.py --cfg_file=./exp/ppo/params.json
```

### Solvent

- Train the ACflow model

``` bash
python scripts/train_model.py --cfg_file=./exp/solvent/params.json
```

- Train the PPO Policy

``` bash
python scripts/train_agent.py --cfg_file=./exp/ppo/params.json --env reg
```

