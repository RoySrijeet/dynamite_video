# Presenting DynaMITe-Video

## Datasets

Store all the datasets in the `datasets` directory. Ensure the paths to images and annotations match the paths specified in [`dynamite_video/utils/paths.py`](dynamite_video/utils/paths.py)

## Installation

TODO

## Environment Variables

Change the following environment variable in [`dynamite_video/.env`](dynamite_video/.env):

```
DYNAMITE_VIDEO_WORKSPACE=""
```

Set it to the path to the git directory

## Training

Default training configurations are specified in `configs/base.yaml`.

Create an experiment folder with a configuration (`.yaml`) file to overwrite the default specifications. 

For example:

```
├── experiments
    ├── experiment_1
        ├── expt_config.yaml
    ├── experiment_2
    
```

## Evaluation

TODO