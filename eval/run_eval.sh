#!/bin/bash

#LLADA eval
#baseline
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
DATASETS=("math" "sudoku" "countdown" "kodcode" "mbpp")
REMASKINGS=("low_confidence" "probabilistic")
for DATASET in "${DATASETS[@]}"; do
  for REMASKING in "${REMASKINGS[@]}"; do
    python -u eval.py \
      --config ../TA-GRPO-d/slurm_scripts/train.yaml \
      --test_model_path "$MODEL_PATH" \
      --dataset "$DATASET" \
      --remasking "$REMASKING" \
      --test_prob_threshold 0.8
  done
done

#for kodcode_train
CODE_DATASETS=("mbpp" "humaneval")
TRAINER_CHECKPOINT_LIST=(
  "../TA-GRPO-d/checkpoints/GSAI-ML/LLaDA-8B-Instruct_code_base_bs12_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/GSAI-ML/LLaDA-8B-Instruct_code_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl32/checkpoint-5000"
)
for DATASET in "${CODE_DATASETS[@]}"; do
  for CHECKPOINT in "${TRAINER_CHECKPOINT_LIST[@]}"; do
    python -u eval.py \
      --config ../TA-GRPO-d/slurm_scripts/train.yaml \
      --trainer_checkpoint_path "$CHECKPOINT" \
      --test_model_path "$MODEL_PATH" \
      --test_dataset "$DATASET" \
      --test_prob_threshold 0.8
  done
done

#for math,sudoku,countdown
TRAINER_CHECKPOINT_LIST=(
  "../TA-GRPO-d/checkpoints/GSAI-ML/LLaDA-8B-Instruct_countdown_base_bs12_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/GSAI-ML/LLaDA-8B-Instruct_countdown_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/GSAI-ML/LLaDA-8B-Instruct_math_base_bs12_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/GSAI-ML/LLaDA-8B-Instruct_math_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/GSAI-ML/LLaDA-8B-Instruct_sudoku_base_bs12_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/GSAI-ML/LLaDA-8B-Instruct_sudoku_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl32/checkpoint-5000"
)
for CHECKPOINT in "${TRAINER_CHECKPOINT_LIST[@]}"; do
  python -u eval.py \
    --config ../TA-GRPO-d/slurm_scripts/train.yaml \
    --trainer_checkpoint_path "$CHECKPOINT" \
    --test_model_path "$MODEL_PATH" \
    --test_prob_threshold 0.8
done


# DREAM eval

#baseline
MODEL_PATH=Dream-org/Dream-v0-Instruct-7B
DATASETS=("math" "sudoku" "countdown" "kodcode" "mbpp") # )
REMASKINGS=("low_confidence" "probabilistic")

for DATASET in "${DATASETS[@]}"; do
  for REMASKING in "${REMASKINGS[@]}"; do
    python -u eval.py \
      --config ../TA-GRPO-d/slurm_scripts/train_dream.yaml \
      --test_model_path "$MODEL_PATH" \
      --dataset "$DATASET" \
      --remasking "$REMASKING" \
      --test_prob_threshold 0.8
  done
done

#for kodcode train
CODE_DATASETS=("mbpp" "humaneval")
TRAINER_CHECKPOINT_LIST=(
  "../TA-GRPO-d/checkpoints/Dream-org/Dream-v0-Instruct-7B_code_base_bs12_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/Dream-org/Dream-v0-Instruct-7B_code_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl32/checkpoint-5000"
)
for DATASET in "${CODE_DATASETS[@]}"; do
  for CHECKPOINT in "${TRAINER_CHECKPOINT_LIST[@]}"; do
    python -u eval.py \
      --config ../TA-GRPO-d/slurm_scripts/train_dream.yaml \
      --trainer_checkpoint_path "$CHECKPOINT" \
      --test_model_path "$MODEL_PATH" \
      --test_dataset "$DATASET" \
      --test_prob_threshold 0.8
  done
done

#for math,sudoku,countdown
TRAINER_CHECKPOINT_LIST=(
  "../TA-GRPO-d/checkpoints/Dream-org/Dream-v0-Instruct-7B_countdown_base_bs12_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/Dream-org/Dream-v0-Instruct-7B_countdown_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/Dream-org/Dream-v0-Instruct-7B_math_base_bs12_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/Dream-org/Dream-v0-Instruct-7B_math_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/Dream-org/Dream-v0-Instruct-7B_sudoku_base_bs12_bl32/checkpoint-5000"
  "../TA-GRPO-d/checkpoints/Dream-org/Dream-v0-Instruct-7B_sudoku_base_bs12_probabilistic_remasking_probthresh0.7_reward_ema_zscore_dp_bl32/checkpoint-5000"
)

for CHECKPOINT in "${TRAINER_CHECKPOINT_LIST[@]}"; do
  python -u eval.py \
    --config ../TA-GRPO-d/slurm_scripts/train_dream.yaml \
    --trainer_checkpoint_path "$CHECKPOINT" \
    --test_model_path "$MODEL_PATH" \
    --test_prob_threshold 0.8
done
