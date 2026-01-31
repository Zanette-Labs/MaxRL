#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
####### SBATCH --qos=earlybird_qos
####### SBATCH --partition=flame-earlybirds
#SBATCH --qos=preempt_qos
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-o5-32,babel-s5-24,babel-m5-32,babel-p5-16,babel-q5-32,babel-q5-24,babel-n5-28,babel-q5-20,babel-q5-16,babel-w9-28,babel-o9-32,babel-q9-16,babel-n5-32,babel-m5-28,babel-w9-20,babel-v9-32,babel-v9-24,babel-n5-20
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-0

source /home/ftajwar/.bashrc
source /home/ftajwar/anaconda3/etc/profile.d/conda.sh
conda activate paprika
cd /home/ftajwar/exploration/slurm_outputs

export NCCL_P2P_DISABLE=1
unset ROCR_VISIBLE_DEVICES

# python -m verl.cifar10_experiments.pytorch_cifar10_experiments \
#     --wandb \
#     --no-amp \
#     --model-type cnn \
#     --model-depth 56 \
#     --data-dir /home/ftajwar/data/cifar100 \
#     --dataset-name cifar100 \
#     --checkpoint-dir /data/user_data/ftajwar/cifar100_checkpoints \
#     --wandb_runname cross_entropy \
#     --wandb-project small_cnn_cifar100_bs_and_num_rollout_sweep \

python -m verl.cifar10_experiments.pytorch_cifar10_rl_experiments \
    --wandb \
    --no-amp \
    --model-depth 20 \
    --model-type cnn \
    --data-dir /home/ftajwar/data/cifar100 \
    --dataset-name cifar100 \
    --checkpoint-dir /data/user_data/ftajwar/cifar100_checkpoints \
    --wandb_runname regular_reinforce_rollouts_131072 \
    --wandb-project small_cnn_cifar100_bs_and_num_rollout_sweep \
    --num_samples_per_example 131072 \
    --advantage-type reinforce_with_baseline \

# torchrun --nproc_per_node=1 ../verl/cifar10_experiments/pytorch_cifar10_rl_experiments_multi_gpu.py \
#     --wandb \
#     --no-amp \
#     --model-depth 20 \
#     --model-type cnn \
#     --data-dir /home/ftajwar/data/cifar100 \
#     --dataset-name cifar100 \
#     --checkpoint-dir /data/user_data/ftajwar/cifar100_checkpoints \
#     --wandb_runname grpo_rollouts_32 \
#     --wandb-project small_cnn_cifar100_bs_and_num_rollout_sweep \
#     --num_samples_per_example 32 \
#     --advantage-type grpo \