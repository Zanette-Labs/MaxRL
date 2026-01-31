#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu_qos
#SBATCH --account=rsalakhu
#SBATCH --job-name=train
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=1700G
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-0

####### aba !/bin/bash
####### SBATCH --job-name=word
###### SBATCH --time=2-00:00:00
###### SBATCH --cpus-per-task=128
###### SBATCH --gpus-per-node=8
###### SBATCH --nodes=1
###### SBATCH --mem=1000G
###### SBATCH --partition=preempt
###### SBATCH --mail-user=ftajwar@cs.cmu.edu
###### SBATCH --mail-type=END,FAIL
###### SBATCH --array=0-0

cd /home/ftajwar
source .bashrc
conda activate paprika

cd /project/flame/ftajwar
rm -rf tmp
mkdir tmp
cd tmp

unset ROCR_VISIBLE_DEVICES

unset ROCR_VISIBLE_DEVICES

export FULL_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16

# Number of rollouts
export NUM_PER_PROMPT_ROLLOUTS=32

# prompt and response length cutoff
export MAX_RESPONSE_LENGTH=3072
export MAX_PROMPT_LENGTH=1024

# Other hyperparameters
export LEARNING_RATE=1e-6
export KL_COEFF=0.001

# RL with ground truth hyperparams
export REWARD_MANAGER='naive'

####
# Change any hyperparams you need, by adding another line here
# For example, if you uncomment the following line
# LEARNING_RATE=3e-7
# the it will set the learning rate to 3e-7 instead of the default value

PER_GPU_MINI_BATCH_SIZE=8
NUM_PER_PROMPT_ROLLOUTS_VALIDATION=32
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=12000
MAX_TRAJECTORY_LENGTH=3900

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2

echo "This is the per-GPU mini batch size: $PER_GPU_MINI_BATCH_SIZE"
echo "This is the Maximum response length: $MAX_RESPONSE_LENGTH"

# TRAIN_DATASET_PATH=/home/ftajwar/data_for_paprika/math_12k_in_instruct_format/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data_for_paprika/math_12k_in_instruct_format/test.parquet
TRAIN_DATASET_PATH=/home/ftajwar/data_for_logp/dapo_subsampled/train.parquet
TEST_DATASET_PATH=/home/ftajwar/data_for_logp/dapo_subsampled/test.parquet

TOTAL_EPOCHS=100

# Set model path
# MODEL_PATH=ftajwar/paprika_Meta-Llama-3.1-8B-Instruct     
# MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
# MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
# MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
MODEL_PATH=Qwen/Qwen2.5-Math-1.5B
# MODEL_PATH=zwhe99/Qwen2.5-7B-orz
# MODEL_PATH=Qwen/Qwen2.5-Math-7B

ADVANTAGE_ESTIMATOR=grpo
PROJECT_NAME=log_p_subsampled_DAPO_Qwen2.5-Math-1.5B
EXPERIMENT_NAME=grpo_with_variance_normalization

VALIDATION_DATA_DIR=/home/ftajwar/online_paprika/validation_${EXPERIMENT_NAME}
ROLLOUT_DATA_DIR=/home/ftajwar/online_paprika/training_${EXPERIMENT_NAME}

# path to save checkpoints
CHECKPOINT_SAVE_PATH=/data/user_data/ftajwar/self_labeling_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
rm -rf $CHECKPOINT_SAVE_PATH

export VLLM_ATTENTION_BACKEND=FLASHINFER
export SEED=69

python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-Math-1.5B')"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADVANTAGE_ESTIMATOR \
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
    trainer.test_freq=35 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    ray_init.ray_dir="/tmp" $@