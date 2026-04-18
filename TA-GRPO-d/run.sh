#!/bin/bash
export LOGDIR=checkpoints
mkdir -p $LOGDIR

DATASET=("countdown" "sudoku") 
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
NUM_ITER=12 # number of policy gradient inner updates iterations
BLOCK_LENGTH=32
PROB_THRESHOLD=0.7

for dataset in "${DATASET[@]}"; do
    echo "=== Running diffu_grpo_train.py for dataset: ${dataset} ==="

    RUN_NAME=${MODEL_PATH}_${dataset}_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl${BLOCK_LENGTH}
    python -u diffu_grpo_train.py \
        --config slurm_scripts/train.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $dataset \
        --run_name $RUN_NAME \
        --max_steps 5000 \
        --block_length $BLOCK_LENGTH \
        --output_dir checkpoints/$RUN_NAME \
        --use_trajectory_aware_rewards true \
        --remasking probabilistic \
        --prob_threshold 0.7 \

    RUN_NAME=${MODEL_PATH}_${dataset}_base_bs12_bl${BLOCK_LENGTH}
    python -u diffu_grpo_train.py \
        --config slurm_scripts/train.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $dataset \
        --run_name $RUN_NAME \
        --max_steps 5000 \
        --block_length $BLOCK_LENGTH \
        --output_dir checkpoints/$RUN_NAME
done

DATASET=("code" "math")
MATH_CODE_DELTA=0.5

for dataset in "${DATASET[@]}"; do
    echo "=== Running diffu_grpo_train.py for dataset: ${dataset} ==="

    RUN_NAME=${MODEL_PATH}_${dataset}_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl${BLOCK_LENGTH}
    python -u diffu_grpo_train.py \
        --config slurm_scripts/train.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $dataset \
        --run_name $RUN_NAME \
        --max_steps 5000 \
        --block_length $BLOCK_LENGTH \
        --output_dir checkpoints/$RUN_NAME \
        --use_trajectory_aware_rewards true \
        --remasking probabilistic \
        --prob_threshold 0.7 \
        --delta_scale $MATH_CODE_DELTA 

    RUN_NAME=${MODEL_PATH}_${dataset}_base_bs12_bl${BLOCK_LENGTH}
    python -u diffu_grpo_train.py \
        --config slurm_scripts/train.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $dataset \
        --run_name $RUN_NAME \
        --max_steps 5000 \
        --block_length $BLOCK_LENGTH \
        --output_dir checkpoints/$RUN_NAME
done


DATASET=("countdown" "sudoku")
# MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
MODEL_PATH=Dream-org/Dream-v0-Instruct-7B
NUM_ITER=12 # number of policy gradient inner updates iterations
BLOCK_LENGTH=32
PROB_THRESHOLD=0.7

for dataset in "${DATASET[@]}"; do
    echo "=== Running diffu_grpo_train.py for dataset: ${dataset} ==="

    RUN_NAME=${MODEL_PATH}_${dataset}_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl${BLOCK_LENGTH}
    python -u diffu_grpo_train.py \
        --config slurm_scripts/train_dream.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $dataset \
        --run_name $RUN_NAME \
        --max_steps 5000 \
        --block_length $BLOCK_LENGTH \
        --output_dir checkpoints/$RUN_NAME \
        --use_trajectory_aware_rewards true \
        --remasking probabilistic \
        --prob_threshold 0.7 \

    RUN_NAME=${MODEL_PATH}_${dataset}_base_bs12_bl${BLOCK_LENGTH}
    python -u diffu_grpo_train.py \
        --config slurm_scripts/train_dream.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $dataset \
        --run_name $RUN_NAME \
        --max_steps 5000 \
        --block_length $BLOCK_LENGTH \
        --output_dir checkpoints/$RUN_NAME
done

DATASET=("code" "math")
MATH_CODE_DELTA=0.5

for dataset in "${DATASET[@]}"; do
    echo "=== Running diffu_grpo_train.py for dataset: ${dataset} ==="

    RUN_NAME=${MODEL_PATH}_${dataset}_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl${BLOCK_LENGTH}
    python -u diffu_grpo_train.py \
        --config slurm_scripts/train_dream.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $dataset \
        --run_name $RUN_NAME \
        --max_steps 5000 \
        --block_length $BLOCK_LENGTH \
        --output_dir checkpoints/$RUN_NAME \
        --use_trajectory_aware_rewards true \
        --remasking probabilistic \
        --prob_threshold 0.7 \
        --delta_scale $MATH_CODE_DELTA 

    RUN_NAME=${MODEL_PATH}_${dataset}_base_bs12_bl${BLOCK_LENGTH}
    python -u diffu_grpo_train.py \
        --config slurm_scripts/train_dream.yaml \
        --model_path $MODEL_PATH \
        --num_iterations $NUM_ITER \
        --dataset $dataset \
        --run_name $RUN_NAME \
        --max_steps 5000 \
        --block_length $BLOCK_LENGTH \
        --output_dir checkpoints/$RUN_NAME
done
