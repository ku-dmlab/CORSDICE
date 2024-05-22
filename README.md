# Redeeming Valid Stationary Distribution Correction from Semi-gradient DICE

This repository is the official implementation of **Redeeming Valid Stationary Distribution Correction from Semi-gradient DICE**. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train_evaluation.py --env_name <env_name> --config configs/mujoco_config.
```

You can find the list of environments in `environment.py` file.

### Reproduction

To reproduce the experiment on continuous environment, please run the following commands.

```train
python train_evaluation.py \
    --env_name <env_name> \
    --config configs/mujoco_config \
    --divergence SoftChi \
    --initial_lambda 1.0 \
    --alpha <0.01, 0.1, 1.0, 10.0>,
    --cost_ub 40.0 \
    --gradient_penalty_coeff 0.01 \
    --lr_ratio 0.1 \
    --seed <42, 420, 4200>
```

Please note that reproducing off-policy evaluation is not available, due to incompatible
API between different versions of Gym.

## Evaluation

You can evaluate the model with the following command.

```train
python train_evaluation.py --env_name <env_name> --ckpt_dir <checkpoint> --config configs/mujoco_config.
```
