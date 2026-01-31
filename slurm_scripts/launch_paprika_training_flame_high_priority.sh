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


cd /home/ftajwar
source .bashrc
conda activate paprika

cd /project/flame/ftajwar
rm -rf tmp
mkdir tmp
cd tmp

unset ROCR_VISIBLE_DEVICES


export FULL_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16

# Number of rollouts
export NUM_PER_PROMPT_ROLLOUTS=256
export NUM_ON_POLICY_ROLLOUTS_PER_PROMPT=256
export NUM_OFF_POLICY_ROLLOUTS_PER_PROMPT=0

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

PER_GPU_MINI_BATCH_SIZE=4
NUM_PER_PROMPT_ROLLOUTS_VALIDATION=32
MAX_MODEL_LEN=32000  
MAX_NUM_BATCHED_TOKENS=32000
MAX_TRAJECTORY_LENGTH=12000

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2

ENTROPY_COEFF=0.0001
# LOSS_AGG_MODE=seq-mean-token-sum-norm
LOSS_AGG_MODE=token-mean
LOSS_TYPE=grpo
ADVANTAGE_ESTIMATOR=p_normalization

echo "This is the per-GPU mini batch size: $PER_GPU_MINI_BATCH_SIZE"
echo "This is the Maximum response length: $MAX_RESPONSE_LENGTH"

# Set training and test data path here
# TRAIN_DATASET_PATH="['/home/ftajwar/data/wordle/train.parquet','/home/ftajwar/data/wordle_modified/train.parquet']"

# TRAIN_DATASET_PATH=/home/ftajwar/data_for_paprika/mastermind_extended/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data_for_paprika/mastermind_extended/test.parquet

TRAIN_DATASET_PATH=/home/ftajwar/data_for_paprika/wordle_qwen3/train.parquet
TEST_DATASET_PATH=/home/ftajwar/data_for_paprika/wordle_qwen3/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data_for_paprika/wordle/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data_for_paprika/wordle/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data_for_paprika/math_12k_in_paprika_format/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data_for_paprika/aime_2024_in_paprika_format/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data/wordle_qwen3/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/wordle_qwen3/train.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data/countdown_in_paprika_format/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/countdown_in_paprika_format/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data/math_12k_in_paprika_format/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/math_12k_in_paprika_format/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data/twenty_questions/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/twenty_questions_num_test_1/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data/cellular_automata/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/cellular_automata/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data/mastermind/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/mastermind/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data/wordle/train.parquet
# TEST_DATASET_PATH="['/home/ftajwar/data/wordle/test.parquet','/home/ftajwar/data/cellular_automata/test.parquet','/home/ftajwar/data/mastermind/test.parquet']"

# TRAIN_DATASET_PATH=/home/ftajwar/data/battleship/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/battleship/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data/minesweeper/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/minesweeper/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data_for_paprika/mastermind_extended/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data_for_paprika/mastermind_extended/test.parquet


# Total Train EPOCHS
TOTAL_EPOCHS=1

# VALIDATION sampling params
VALIDATION_TEMPERATURE=1.0
VALIDATION_TOP_P=1.0
VALIDATION_TOP_K=-1
VALIDATION_MIN_P=0.0

# WANDB logging
PROJECT_NAME=qwen3_wordle_sampling_experiments
EXPERIMENT_NAME=temp_${VALIDATION_TEMPERATURE}_top_p_${VALIDATION_TOP_P}_top_k_${VALIDATION_TOP_K}_min_p_${VALIDATION_MIN_P}

VALIDATION_DATA_DIR=/project/flame/ftajwar/online_paprika_rollouts/${PROJECT_NAME}/validation_${EXPERIMENT_NAME}
ROLLOUT_DATA_DIR=/project/flame/ftajwar/online_paprika_rollouts/${PROJECT_NAME}/training_${EXPERIMENT_NAME}


# Set model path
# MODEL_PATH=ftajwar/paprika_Meta-Llama-3.1-8B-Instruct      
# MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-Math-1.5B
# MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
# MODEL_PATH=zwhe99/Qwen2.5-7B-orz
# MODEL_PATH=Qwen/Qwen2.5-3B
# MODEL_PATH=Qwen/Qwen3-1.7B
# MODEL_PATH=Qwen/Qwen3-4B
MODEL_PATH=Qwen/Qwen3-4B
# MODEL_PATH=Qwen/Qwen2.5-7B

# MODEL_PATH=/project/flame/ftajwar/online_paprika_checkpoints/qwen3_4B_wordle_sft_checkpoint
# POLICY_MODEL_NAME=Qwen/Qwen3-4B

POLICY_MODEL_NAME=$MODEL_PATH

# LOG_PROB_SAVE_PATH=/project/flame/ftajwar/paprika_offpolicy_data/qwen3_8B_wordle.pt
REPLAY_BUFFER_LOCATION=/project/flame/ftajwar/paprika_offpolicy_data/qwen3_8B_wordle.pt


# path to save checkpoints
CHECKPOINT_SAVE_PATH=/tmp/online_paprika_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
rm -rf $CHECKPOINT_SAVE_PATH

export VLLM_ATTENTION_BACKEND=XFORMERS
export SEED=69

python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen3-4B')"


python3 -m verl.paprika.main_paprika \
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
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=$VALIDATION_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VALIDATION_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.top_k=$VALIDATION_TOP_K \
    actor_rollout_ref.rollout.val_kwargs.min_p=$VALIDATION_MIN_P \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.policy_model_name=$POLICY_MODEL_NAME \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=$KL_COEFF \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.no_advantage_weighting_on_off_policy_data=False \
    reward_model.reward_manager=$REWARD_MANAGER \
    trainer.loss_type=$LOSS_TYPE \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.use_replay_buffer=True \
    trainer.generate_and_save_validation_log_probs=False \
    trainer.num_onpolicy_rollouts_per_task_group=$NUM_ON_POLICY_ROLLOUTS_PER_PROMPT \
    trainer.num_offpolicy_rollouts_per_task_group=$NUM_OFF_POLICY_ROLLOUTS_PER_PROMPT \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CHECKPOINT_SAVE_PATH \
    trainer.n_gpus_per_node=8  \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=10 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    ray_init.ray_dir="/tmp" $@

# trainer.path_to_load_replay_buffer=$REPLAY_BUFFER_LOCATION \
# trainer.validation_data_dir=$VALIDATION_DATA_DIR \

# python3 -m verl.paprika.main_paprika \
#     algorithm.adv_estimator=grpo \
#     data.train_files=$TRAIN_DATASET_PATH \
#     data.val_files=$TEST_DATASET_PATH \
#     data.train_batch_size=$FULL_BATCH_SIZE \
#     data.max_prompt_length=$MAX_PROMPT_LENGTH \
#     data.max_response_length=$MAX_RESPONSE_LENGTH \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=$MODEL_PATH \
#     actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.actor.kl_loss_coef=$KL_COEFF \
#     actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
#     actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
#     actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
#     actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm_multiturn \
#     actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
#     actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
#     actor_rollout_ref.rollout.trajectory_length=$MAX_TRAJECTORY_LENGTH \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
#     actor_rollout_ref.rollout.n=$NUM_PER_PROMPT_ROLLOUTS \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.rollout.val_kwargs.n=$NUM_PER_PROMPT_ROLLOUTS_VALIDATION \
#     actor_rollout_ref.rollout.val_kwargs.do_sample=True \
#     actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
#     actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
#     actor_rollout_ref.rollout.val_kwargs.top_k=20 \
#     actor_rollout_ref.rollout.val_kwargs.min_p=0.3 \
#     actor_rollout_ref.rollout.multi_turn.enable=True \
#     actor_rollout_ref.rollout.policy_model_name=$POLICY_MODEL_NAME \
#     algorithm.use_kl_in_reward=False \
#     algorithm.kl_penalty=low_var_kl \
#     algorithm.kl_ctrl.kl_coef=$KL_COEFF \
#     algorithm.norm_adv_by_std_in_grpo=True \
#     reward_model.reward_manager=$REWARD_MANAGER \
#     trainer.critic_warmup=0 \
#     trainer.val_before_train=True \
#     trainer.val_only=False \
#     trainer.use_replay_buffer=False \
#     trainer.generate_and_save_validation_log_probs=False \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name=$PROJECT_NAME \
#     trainer.experiment_name=$EXPERIMENT_NAME \
#     trainer.default_local_dir=$CHECKPOINT_SAVE_PATH \
#     trainer.n_gpus_per_node=8  \
#     trainer.nnodes=1 \
#     trainer.save_freq=-1 \
#     trainer.max_actor_ckpt_to_keep=1 \
#     trainer.max_critic_ckpt_to_keep=1 \
#     trainer.test_freq=10 \
#     trainer.total_epochs=$TOTAL_EPOCHS \
#     ray_init.ray_dir="/tmp" $@

#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.model.path=$MODEL_PATH \
#     critic.model.enable_gradient_checkpointing=True \
#     critic.ppo_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
#     critic.model.fsdp_config.param_offload=False \
#     critic.model.fsdp_config.optimizer_offload=False \
# trainer.path_to_save_validation_log_prob=$LOG_PROB_SAVE_PATH \
# trainer.validation_data_dir=$VALIDATION_DATA_DIR \


# convert and save condensed checkpoint

# FINAL_CHECKPOINT_PATH=/tmp/online_paprika_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_282/actor
# CONDENSED_CHECKPOINT_PATH=/project/flame/ftajwar/online_paprika_checkpoints/qwen3_4B_wordle_fully_offpolicy_empo_checkpoint

# cd /home/ftajwar/exploration/scripts

# python model_merger.py merge --backend fsdp --local_dir $FINAL_CHECKPOINT_PATH --target_dir $CONDENSED_CHECKPOINT_PATH

# rm -rf $FINAL_CHECKPOINT_PATH

# python3 -m verl.paprika.main_paprika \
#     algorithm.adv_estimator=grpo \
#     data.train_files=$TRAIN_DATASET_PATH \
#     data.val_files=$TEST_DATASET_PATH \
#     data.train_batch_size=$FULL_BATCH_SIZE \
#     data.max_prompt_length=$MAX_PROMPT_LENGTH \
#     data.max_response_length=$MAX_RESPONSE_LENGTH \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=$MODEL_PATH \
#     actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.actor.kl_loss_coef=$KL_COEFF \
#     actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
#     actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
#     actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
#     actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm_multiturn \
#     actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
#     actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
#     actor_rollout_ref.rollout.trajectory_length=$MAX_TRAJECTORY_LENGTH \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
#     actor_rollout_ref.rollout.n=$NUM_PER_PROMPT_ROLLOUTS \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.rollout.val_kwargs.n=$NUM_PER_PROMPT_ROLLOUTS_VALIDATION \
#     actor_rollout_ref.rollout.val_kwargs.do_sample=True \
#     actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
#     actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
#     actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
#     actor_rollout_ref.rollout.val_kwargs.min_p=0.0 \
#     actor_rollout_ref.rollout.multi_turn.enable=True \
#     actor_rollout_ref.rollout.policy_model_name=$POLICY_MODEL_NAME \
#     algorithm.use_kl_in_reward=False \
#     algorithm.kl_penalty=low_var_kl \
#     algorithm.kl_ctrl.kl_coef=$KL_COEFF \
#     algorithm.norm_adv_by_std_in_grpo=True \
#     reward_model.reward_manager=$REWARD_MANAGER \
#     trainer.critic_warmup=0 \
#     trainer.val_before_train=True \
#     trainer.val_only=True \
#     trainer.use_replay_buffer=True \
#     trainer.generate_and_save_validation_log_probs=True \
#     trainer.path_to_save_validation_log_prob=$LOG_PROB_SAVE_PATH \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name=$PROJECT_NAME \
#     trainer.experiment_name=$EXPERIMENT_NAME \
#     trainer.default_local_dir=$CHECKPOINT_SAVE_PATH \
#     trainer.n_gpus_per_node=8  \
#     trainer.nnodes=1 \
#     trainer.save_freq=10 \
#     trainer.max_actor_ckpt_to_keep=1 \
#     trainer.max_critic_ckpt_to_keep=1 \
#     trainer.test_freq=10 \
#     trainer.total_epochs=$TOTAL_EPOCHS \
#     trainer.validation_data_dir=$VALIDATION_DATA_DIR \
#     ray_init.ray_dir="/tmp" $@