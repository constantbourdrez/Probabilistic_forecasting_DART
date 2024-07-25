# DART
Probabilistic forecasting of dengue fever

## Load a pretrained model

There is no script yet to load esaily trained model so just run this lines in lightning_module.py

```python
from model_pipe.utils import *
model = load_model_from_previous_state(config_path = path_to_config, checkpoint_path = path_to_checkpoint, hparams_path = path_to_hparams)
```
## Launch a train

Run this command in model_pipe:

```
python train.py --config config.yaml
```
## Vizualize training metrics

```
cd model_pipe
tensorboard --logdir=lightning_logs/
```
## Launch testing

```
cd model_pipe
python test.py --config config.yaml --sample_steps 1   --beta_schedules uniform  --num_samples 5  --chk_path lightning_logs/version_23/checkpoints/epoch=5-step=1140.ckpt --hparams_path lightning_logs/version_23/hparams.yaml
```
