# TA-GRPO-d: Trajectory-Aware GRPO for Optimizing Denoising Trajectories in Diffusion LLMs

This repository contains an implementation of TA-GRPO-d, a reinforcement learning method for improving denoising trajectories in discrete diffusion language models.

This implementation is based on the public `d1` repository:

- https://github.com/dllm-reasoning/d1

The method combines:

- confidence-gated denoising for adaptive unmasking
- group-relative rewards over multiple sampled trajectories
- trajectory-aware reward shaping with intermediate rewards and unmasking-time weighting

## Important Warning

Current public support is limited.

- Standalone batched generation and evaluation are only considered reliable at `batch_size=1`.
- Single-process loading with `device_map="auto"` is the only multi-GPU mode currently expected to work.
- Data-parallel multi-GPU execution is not officially supported in the current release.

Why this limitation exists:

- The generation path currently does not use attention masks during generation. When prompts of different lengths are padded together in one batch, padded tokens can affect generation behavior.
- In `TA-GRPO-d/diffu_grpo_trainer.py`, the reward pipeline gathers rewards across processes and then restores local tensors using a rank-based slice. This assumes a tightly aligned per-rank layout and is brittle for data-parallel distributed execution.
- The trajectory-aware reward path further mixes gathered rewards with per-rank local trajectory buffers, so the current implementation is not yet safe for data-parallel multi-process execution.
- Some output-directory and logging writes are also not rank-safe in a multi-process launch.

A patch for proper batched generation and data-parallel multi-GPU support will be released in a future update.

## Repository Layout

- `TA-GRPO-d/`: training code, configs, reward functions, and experiment scripts
- `eval/`: evaluation, generation saving, and result parsing scripts
- `dataset/`: local dataset files used by the repository
- `env.yml`: environment specification

## Environment

We recommend creating the environment from `env.yml`.

```bash
conda env create -f env.yml
conda activate <your-env-name>
```

You will also need access to the base models used in the experiments, including:

- `GSAI-ML/LLaDA-8B-Instruct`
- `Dream-org/Dream-v0-Instruct-7B`

## Paper Experiment Scripts

For this release, the following scripts are the paper-aligned experiment pipeline used in this codebase:

- `TA-GRPO-d/run.sh`: training runs
- `eval/run_eval.sh`: evaluation runs
- `eval/parse_and_get_acc.py`: result parsing and accuracy aggregation

If you want to reproduce the reported training and evaluation flow in this repository, please start from those scripts.

## Reimplementation Note

This release is a reimplementation that tracks the evolving `d1` codebase rather than an exact frozen copy of the original paper snapshot.

Because of that, some low-level details can differ slightly from the paper implementation, especially in areas such as:

- reward parsing rules
- reward format handling
- evaluation-side parsing behavior

In our checks, the overall performance trend is very close to the paper results, although small numerical differences can still appear because of those implementation-level details.

## Training

Run the main experiment script:

```bash
cd TA-GRPO-d
bash run.sh
```

The script includes runs for:

- `LLaDA-8B-Instruct`
- `Dream-v0-Instruct-7B`
- `countdown`, `sudoku`, `math`, and `code`

The paper comparison in this repository is controlled mainly by:

- `--remasking probabilistic`
- `--use_trajectory_aware_rewards true`

compared against the corresponding baseline runs without those settings.

## Evaluation

Run generation and evaluation:

```bash
cd eval
bash run_eval.sh
python parse_and_get_acc.py
```

`run_eval.sh` saves generation files to the results directory, and `parse_and_get_acc.py` aggregates task accuracy and average diffusion steps from those saved outputs.


## Notes

- The repository `.gitignore` already excludes the main generated artifacts such as checkpoints, outputs, temporary files, execution files, logs, and result files.
- If you adapt the scripts for new experiments, please double-check paths, checkpoint names, and model availability before launching long runs.
- This repository should be viewed as a faithful reimplementation with very similar empirical behavior, not as a byte-identical release of the original internal training snapshot.

## Acknowledgement

This implementation was built on top of the public `d1(https://github.com/dllm-reasoning/d1)` repository and adapted for TA-GRPO-d experiments.


```
@article{zhao2025d1,
  title={d1: Scaling reasoning in diffusion large language models via reinforcement learning},
  author={Zhao, Siyan and Gupta, Devaansh and Zheng, Qinqing and Grover, Aditya},
  journal={arXiv preprint arXiv:2504.12216},
  year={2025}
}
```

## Citation

If you use this repository, please cite the corresponding paper once the final bibliographic information is available.
