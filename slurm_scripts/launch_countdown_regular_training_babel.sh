#!/bin/bash

#SBATCH --job-name=math
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --exclude=babel-4-1,babel-3-5,babel-8-13,babel-14-13,babel-12-29,babel-13-9,babel-13-21,babel-13-13,babel-3-17,babel-10-9,babel-10-13,babel-10-5,babel-7-1,babel-2-25,babel-14-1,babel-6-29,babel-13-1,babel-8-9,babel-8-5,babel-11-25,babel-7-9,babel-14-37,babel-4-13
#SBATCH --array=0-0

source /home/ftajwar/.bashrc
source /home/ftajwar/anaconda3/etc/profile.d/conda.sh
conda activate paprika
cd /home/ftajwar/exploration/slurm_outputs

export NCCL_P2P_DISABLE=1
unset ROCR_VISIBLE_DEVICES

export FULL_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16

# Number of rollouts
export NUM_PER_PROMPT_ROLLOUTS=8

# prompt and response length cutoff
export MAX_RESPONSE_LENGTH=1024
export MAX_PROMPT_LENGTH=1024

# Other hyperparameters
export LEARNING_RATE=5e-7
export KL_COEFF=0.001

# RL with ground truth hyperparams
export REWARD_MANAGER='naive'

####
# Change any hyperparams you need, by adding another line here
# For example, if you uncomment the following line
# LEARNING_RATE=3e-7
# the it will set the learning rate to 3e-7 instead of the default value

PER_GPU_MINI_BATCH_SIZE=1
NUM_PER_PROMPT_ROLLOUTS_VALIDATION=4
MAX_MODEL_LEN=5000
MAX_NUM_BATCHED_TOKENS=12000
MAX_TRAJECTORY_LENGTH=2048

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2

echo "This is the per-GPU mini batch size: $PER_GPU_MINI_BATCH_SIZE"
echo "This is the Maximum response length: $MAX_RESPONSE_LENGTH"

# Set training and test data path here
# TRAIN_DATASET_PATH="['/home/ftajwar/data/wordle/train.parquet','/home/ftajwar/data/wordle_modified/train.parquet']"

# TRAIN_DATASET_PATH=/home/ftajwar/data/countdown_instruct_format/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/countdown_instruct_format/test.parquet

TRAIN_DATASET_PATH=/home/ftajwar/data/math_12k_instruct_format/train.parquet
TEST_DATASET_PATH=/home/ftajwar/data/math_12k_instruct_format/test.parquet

# WANDB logging
PROJECT_NAME=Debug_MATH_12k_Qwen2.5-3B
EXPERIMENT_NAME=regular_training

VALIDATION_DATA_DIR=/home/ftajwar/online_paprika/validation_${EXPERIMENT_NAME}
ROLLOUT_DATA_DIR=/home/ftajwar/online_paprika/training_${EXPERIMENT_NAME}

# Total Train EPOCHS
TOTAL_EPOCHS=1

# Set model path
# MODEL_PATH=ftajwar/paprika_Meta-Llama-3.1-8B-Instruct      
# MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-Math-1.5B
# MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
# MODEL_PATH=zwhe99/Qwen2.5-7B-orz
MODEL_PATH=Qwen/Qwen2.5-3B

# path to save checkpoints
CHECKPOINT_SAVE_PATH=/data/user_data/ftajwar/self_labeling_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
rm -rf $CHECKPOINT_SAVE_PATH

export VLLM_ATTENTION_BACKEND=XFORMERS
export SEED=69

# python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-3B')"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATASET_PATH \
    data.val_files=$TEST_DATASET_PATH \
    data.train_batch_size=$FULL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEFF \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$NUM_PER_PROMPT_ROLLOUTS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=$NUM_PER_PROMPT_ROLLOUTS_VALIDATION \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=$KL_COEFF \
    reward_model.reward_manager=$REWARD_MANAGER \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CHECKPOINT_SAVE_PATH \
    trainer.n_gpus_per_node=8  \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=25 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.validation_data_dir=$VALIDATION_DATA_DIR \
    ray_init.ray_dir="/tmp" $@

#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.model.path=$MODEL_PATH \
#     critic.model.enable_gradient_checkpointing=True \
#     critic.ppo_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
#     critic.model.fsdp_config.param_offload=False \
#     critic.model.fsdp_config.optimizer_offload=False \