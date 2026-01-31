#!/bin/bash
#SBATCH --job-name="train"
#SBATCH --account=bdtp-delta-gpu
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpuH200x8
#SBATCH --mem=150G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=8   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --no-requeue
#SBATCH -t 06:00:00


source /u/ftajwar/.bashrc
source /u/ftajwar/anaconda3/etc/profile.d/conda.sh
conda activate paprika
cd /u/ftajwar/exploration/slurm_outputs

unset ROCR_VISIBLE_DEVICES


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 algorithm.use_kl_in_reward=False \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.0 \
 trainer.logger=console \
 trainer.val_before_train=True \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=-1 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log