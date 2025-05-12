#!/bin/bash

DATASET_DIR="/repos/datasets"

# Fast dev run
# python mnist_pl.py --output_grad_norm --fast_dev_run --dataset_dir $DATASET_DIR

# Track gradients
python mnist_pl.py --output_grad_norm --max_epochs 1 --dataset_dir $DATASET_DIR

# Different learning rates
python mnist_pl.py  --learning_rate 0.0001 --max_epochs 1  --dataset_dir $DATASET_DIR
# python mnist_pl.py --learning_rate 0.001 --max_epochs 1  --dataset_dir $DATASET_DIR
python mnist_pl.py --learning_rate 0.01 --max_epochs 1  --dataset_dir $DATASET_DIR

# Different optimizers
# python mnist_pl.py --optimizer "Adam" --max_epochs 1 --dataset_dir $DATASET_DIR
python mnist_pl.py --optimizer "RMSProp" --max_epochs 1 --dataset_dir $DATASET_DIR
python mnist_pl.py --optimizer "AdaGrad" --max_epochs 1 --dataset_dir $DATASET_DIR
