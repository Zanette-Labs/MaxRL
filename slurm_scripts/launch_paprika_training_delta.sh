#!/bin/bash
#SBATCH --job-name="train"
#SBATCH --account=bdtp-delta-gpu
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpuH200x8
#SBATCH --mem=1500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=96   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --no-requeue
#SBATCH -t 48:00:00

source /u/ftajwar/.bashrc
source /u/ftajwar/anaconda3/etc/profile.d/conda.sh
conda activate paprika
cd /u/ftajwar/exploration/slurm_outputs

unset ROCR_VISIBLE_DEVICES

export FULL_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16

# Number of rollouts
export NUM_PER_PROMPT_ROLLOUTS=32

# prompt and response length cutoff
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=1024

# Other hyperparameters
export LEARNING_RATE=5e-7
export KL_COEFF=0.001

# RL with ground truth hyperparams
export REWARD_MANAGER='paprika'

####
# Change any hyperparams you need, by adding another line here
# For example, if you uncomment the following line
# LEARNING_RATE=3e-7
# the it will set the learning rate to 3e-7 instead of the default value

PER_GPU_MINI_BATCH_SIZE=8
NUM_PER_PROMPT_ROLLOUTS_VALIDATION=4
MAX_MODEL_LEN=32000
MAX_NUM_BATCHED_TOKENS=32000
MAX_TRAJECTORY_LENGTH=29000

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2

echo "This is the per-GPU mini batch size: $PER_GPU_MINI_BATCH_SIZE"
echo "This is the Maximum response length: $MAX_RESPONSE_LENGTH"

# Set training and test data path here

# TRAIN_DATASET_PATH="['/u/ftajwar/data/jotto/train.parquet','/u/ftajwar/data/wordle/train.parquet','/u/ftajwar/data/battleship/train.parquet','/u/ftajwar/data/mastermind/train.parquet','/u/ftajwar/data/minesweeper/train.parquet','/u/ftajwar/data/hangman/train.parquet','/u/ftajwar/data/bandit_best_arm_identification/train.parquet','/u/ftajwar/data/cellular_automata/train.parquet']"
# TEST_DATASET_PATH="['/u/ftajwar/data/jotto/test.parquet','/u/ftajwar/data/wordle/test.parquet','/u/ftajwar/data/battleship/test.parquet','/u/ftajwar/data/mastermind/test.parquet','/u/ftajwar/data/minesweeper/test.parquet','/u/ftajwar/data/hangman/test.parquet','/u/ftajwar/data/bandit_best_arm_identification/test.parquet','/u/ftajwar/data/cellular_automata/test.parquet']"

TRAIN_DATASET_PATH=/u/ftajwar/data/mastermind_extended/train.parquet
TEST_DATASET_PATH=/u/ftajwar/data/mastermind_extended/test.parquet

# TRAIN_DATASET_PATH=/u/ftajwar/data/wordle/train.parquet
# TEST_DATASET_PATH=/u/ftajwar/data/wordle/test.parquet

# TRAIN_DATASET_PATH=/u/ftajwar/data/wordle_qwen3/train.parquet
# TEST_DATASET_PATH=/u/ftajwar/data/wordle_qwen3/test.parquet

# Total Train EPOCHS
TOTAL_EPOCHS=1

# Set model path
# MODEL_PATH=Qwen/Qwen3-8B
# MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
# MODEL_PATH=ftajwar/paprika_Meta-Llama-3.1-8B-Instruct 

POLICY_MODEL_NAME=$MODEL_PATH

ENTROPY_COEFF=0.0
# LOSS_AGG_MODE=seq-mean-token-sum-norm
LOSS_AGG_MODE=token-mean

PROJECT_NAME=llama_3_mastermind_off_policy_experiments_first_trial
EXPERIMENT_NAME=on_policy_training_llama_8B
TENSOR_MODEL_PARALLEL_SIZE=1


VALIDATION_DATA_DIR=/u/ftajwar/online_paprika_rollouts/${PROJECT_NAME}/validation_${EXPERIMENT_NAME}
ROLLOUT_DATA_DIR=/u/ftajwar/online_paprika_rollouts/${PROJECT_NAME}/training_${EXPERIMENT_NAME}
# LOG_PROB_SAVE_PATH=/data/user_data/ftajwar/paprika_offpolicy_data/qwen3_4B_data.pt

# path to save checkpoints
CHECKPOINT_SAVE_PATH=/projects/bfgl/ftajwar/online_paprika_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
rm -rf $CHECKPOINT_SAVE_PATH

export VLLM_ATTENTION_BACKEND=XFORMERS
export SEED=69

python3 -c "import transformers; transformers.pipeline('text-generation', model='meta-llama/Meta-Llama-3.1-8B-Instruct')"

python3 -m verl.paprika.main_paprika \
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
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm_multiturn \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.trajectory_length=$MAX_TRAJECTORY_LENGTH \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$NUM_PER_PROMPT_ROLLOUTS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=$NUM_PER_PROMPT_ROLLOUTS_VALIDATION \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.min_p=0.3 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.policy_model_name=$POLICY_MODEL_NAME \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=$KL_COEFF \
    algorithm.norm_adv_by_std_in_grpo=True \
    reward_model.reward_manager=$REWARD_MANAGER \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.use_replay_buffer=False \
    trainer.generate_and_save_validation_log_probs=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CHECKPOINT_SAVE_PATH \
    trainer.n_gpus_per_node=8  \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=50 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.validation_data_dir=$VALIDATION_DATA_DIR \
    ray_init.ray_dir="/tmp" $@

# trainer.validation_data_dir=$VALIDATION_DATA_DIR \
# trainer.path_to_save_validation_log_prob=$LOG_PROB_SAVE_PATH \