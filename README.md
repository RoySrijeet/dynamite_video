# Towards Video Multi-object Interactive Segmentation

## Datasets

Store all the datasets in the `datasets` directory. Ensure the paths to images and annotations match the paths specified in [`dynamite_video/utils/paths.py`](dynamite_video/utils/paths.py)

## Environment Variables

Change the following environment variable in [`dynamite_video/.env`](dynamite_video/.env):

```
DYNAMITE_VIDEO_WORKSPACE=""
```

Set it to the path to the git directory (containing folders `configs/`, `datasets/`, `detectron2/`, `dynamite_video/`, `weights/`)

## Installation

Training and evaluation performed on Python 3.12.3 and CUDA 12. 

### Install dependencies

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install numpy==1.26.4
pip3 install matplotlib pillow opencv-python tqdm einops imgaug wandb timm python-dotenv
```

### Install detectron2

Install in `$DYNAMITE_VIDEO_WORKSPACE` directory:
```
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git checkout v0.6
python -m pip install -e .
```

### Install MSDA

To install MSDA on a system without GPU, manually set TORCH_CUDA_ARCH_LIST corresponding to available GPU.

```
TORCH_CUDA_ARCH_LIST='8.9 8.6 8.0 7.5' FORCE_CUDA=1 python setup.py build install
cd dynamite_video/model/pixel_decoder/ops
sh make.sh
```


## Training

Default training configurations are specified in [`configs/base.yaml`](configs/base.yaml).

Create an experiment folder with a configuration (`.yaml`) file to overwrite the default specifications.

For example:

```
├── experiments
    ├── experiment_1
        ├── expt_config.yaml
    ├── experiment_2
    
```

An example configuration for training is provided in [here](configs/expt_config_sample.yaml). 

Run:

```
python dynamite_video/main.py --num-gpus 4 --num-machines 1 --expt-dir path/to/experiment/directory
```

### Pre-trained weights

Follow instructions in [`weights/README.md`](weights/README.md) to download pre-trained weights.

## Evaluation

An example configuration for evaluation is provided in [here](configs/eval_config_sample.yaml).

Run:

```
python dynamite_video/main.py --eval-only --expt-dir path/to/experiment/directory --eval-config path/to/evaluation/config/yaml
```

Outputs are saved to the experiment directory by default.