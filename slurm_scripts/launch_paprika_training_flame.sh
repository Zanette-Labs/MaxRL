#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=1000G
#SBATCH --partition=preempt
#SBATCH --mail-user=ftajwar@cs.cmu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-0

cd /home/ftajwar
source .bashrc
conda activate paprika

cd /project/flame/ftajwar/tmp

unset ROCR_VISIBLE_DEVICES

# Cluster code, multi node
# Start your cluster

# echo "job is starting on `hostname`"

# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# nodes_array=($nodes)

# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# # if we detect a space character in the head node IP, we'll
# # convert it to an ipv4 address. This step is optional.
# if [[ "$head_node_ip" == *" "* ]]; then
# IFS=' ' read -ra ADDR <<<"$head_node_ip"
# if [[ ${#ADDR[0]} -gt 16 ]]; then
#   head_node_ip=${ADDR[1]}
# else
#   head_node_ip=${ADDR[0]}
# fi
# echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
# fi
# # __doc_head_address_end__

# # __doc_head_ray_start__
# port=6379
# ip_head=$head_node_ip:$port
# export ip_head
# export RAY_ADDRESS=$ip_head
# echo "IP Head: $ip_head"

# echo "Starting HEAD at $head_node"

# ray stop
# srun --nodes=1 --ntasks=1 -w "$head_node" \
#     ray start --head --node-ip-address="$head_node_ip" --port=$port \
#     --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &
# # __doc_head_ray_end__

# # __doc_worker_ray_start__
# # optional, though may be useful in certain versions of Ray < 1.0.
# sleep 10

# # number of nodes other than the head node
# worker_num=$((SLURM_JOB_NUM_NODES - 1))

# for ((i = 1; i <= worker_num; i++)); do
#     node_i=${nodes_array[$i]}
#     echo "Starting WORKER $i at $node_i"
#     srun --nodes=1 --ntasks=1 -w "$node_i" \
#         ray start --address "$ip_head" \
#         --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &
#     sleep 15
# done

unset ROCR_VISIBLE_DEVICES

export FULL_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16

# Number of rollouts
export NUM_PER_PROMPT_ROLLOUTS=32

# prompt and response length cutoff
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=512

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
NUM_PER_PROMPT_ROLLOUTS_VALIDATION=4
MAX_MODEL_LEN=20000
MAX_NUM_BATCHED_TOKENS=20000
MAX_TRAJECTORY_LENGTH=17900

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2

ENTROPY_COEFF=0.0

echo "This is the per-GPU mini batch size: $PER_GPU_MINI_BATCH_SIZE"
echo "This is the Maximum response length: $MAX_RESPONSE_LENGTH"

# Set training and test data path here

# TRAIN_DATASET_PATH=/home/ftajwar/data/big_math_filtered_in_paprika_format/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/math_in_paprika_format/test.parquet

# TRAIN_DATASET_PATH=/home/ftajwar/data/math_in_paprika_format/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/math_in_paprika_format/test.parquet

# TRAIN_DATASET_PATH="['/home/ftajwar/data/wordle/train.parquet','/home/ftajwar/data/wordle_modified/train.parquet']"
# TRAIN_DATASET_PATH=/home/ftajwar/data/wordle/train.parquet
# TEST_DATASET_PATH=/home/ftajwar/data/wordle/test.parquet

TRAIN_DATASET_PATH=/home/ftajwar/data/battleship/train.parquet
TEST_DATASET_PATH=/home/ftajwar/data/battleship/test.parquet

# Total Train EPOCHS
TOTAL_EPOCHS=1

# Set model path
# MODEL_PATH=ftajwar/paprika_Meta-Llama-3.1-8B-Instruct     
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
# MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
# MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-Math-7B
# MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
# MODEL_PATH=zwhe99/Qwen2.5-7B-orz

# WANDB logging
# PROJECT_NAME=Wordle
# EXPERIMENT_NAME=off_policy_reinforce
PROJECT_NAME=Llama-3.1-8B-Instruct
EXPERIMENT_NAME=battleship_single_task_on_policy_no_kl_no_entropy

VALIDATION_DATA_DIR=/home/ftajwar/online_paprika/validation_${EXPERIMENT_NAME}
ROLLOUT_DATA_DIR=/home/ftajwar/online_paprika/training_${EXPERIMENT_NAME}

# path to save checkpoints
CHECKPOINT_SAVE_PATH=/tmp/self_labeling_checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
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
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.min_p=0.3 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
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
    trainer.test_freq=10 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    ray_init.ray_dir="/project/flame/ftajwar/tmp" $@

# actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
# actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
# trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
# trainer.validation_data_dir=$VALIDATION_DATA_DIR \
# critic.optim.lr=1e-5 \
# critic.model.path=ftajwar/paprika_Meta-Llama-3.1-8B-Instruct  \
# critic.model.enable_gradient_checkpointing=False \
# critic.ppo_micro_batch_size_per_gpu=$PER_GPU_MINI_BATCH_SIZE \