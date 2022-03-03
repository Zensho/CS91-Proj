## Getting Started

First, download the source code.
Then, download the dataset "LR_task_with_antisaccade_synchronised_min_hilbert.npz" and"LR_task_with_antisaccade_synchronised_min.npz"in the EEGEyeNet Dataset(https://osf.io/jkrzh/). Then put both datasets into a folder called "data".

### Prerequisites
The required packages can be installed easily with conda:
conda create -n eegeyenet_benchmark python=3.8.5
conda install --file general_requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install --file tensorflow_requirements.txt
conda install transformers
conda install --file standard_ml_requirements.txt

### Recreate results

For the recreation of the EEGEyeNet benchmark results run main.py.
```
python main.py
```
### Transformers-related code

To see the transformers-related code go to transformersEEG.py

### Config for extracted raw data
### config.py
```
config['task'] = 'LR_task'
config['dataset'] = 'antisaccade'
config['preprocessing'] = 'min'
config['feature_extraction'] = True
config['include_ML_models'] = True
config['include_DL_models'] = True
config['include_your_models'] = False
config['include_dummy_models'] = True
config['retrain'] = True
config['save_models'] = True
config['load_experiment_dir'] = path/to/your/model
```

### Obtain shortened raw data
To obtain the shortened version (with 2000 data entries) of the data, 
```
python unzip.py
```