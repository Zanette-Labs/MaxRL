#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
####### SBATCH --qos=earlybird_qos
####### SBATCH --partition=flame-earlybirds
#SBATCH --qos=normal
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
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

export NUM_PER_PROMPT_ROLLOUTS=16

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=p_normalization \
 algorithm.use_kl_in_reward=False \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=32 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 actor_rollout_ref.rollout.n=${NUM_PER_PROMPT_ROLLOUTS} \
 actor_rollout_ref.rollout.val_kwargs.n=1 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.0 \
 trainer.val_before_train=True \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=-1 \
 trainer.test_freq=50 \
 trainer.total_epochs=1 \
 trainer.logger=['console','wandb'] \
 trainer.project_name=gsm8k_qwen0.5_instruct \
 trainer.experiment_name=modified_p_normalization_n_rollouts_${NUM_PER_PROMPT_ROLLOUTS} \
 ray_init.ray_dir=/tmp \