import argparse
import json
import math
import os
import random
import time


import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.serialization import add_safe_globals
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel
from trl import TrlParser, ModelConfig

from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset
from humaneval import HumanevalDataset
from mbpp import MBPPDataset

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TA-GRPO-d")))
from diffu_grpo_trainer import DiffuGRPOTrainer
from diffu_grpo_config import DiffuGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
)
from reward_func_code import (
    kodcode_reward_func
)

from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    set_random_seed,
    get_math_questions,
)

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
    "humaneval": HumanevalDataset,
    "mbpp": MBPPDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(
    trainer,
    dataloader,
    save_processing_dir=None
):
    trainer.model.eval()
    with torch.no_grad():
        total_processed = torch.tensor(0, device=trainer.model.device)
        wall_times = []
        all_generations = []
        device = trainer.model.device

        for batch in tqdm(dataloader):
            start_time = time.time()
            input_ids = batch["input_ids"].to(device)
            gt_answers = batch["answers"]
            questions = batch["questions"]
            prompts = batch["prompts"]
            
            print(f"Input ids shape: {input_ids.shape}, dtype: {input_ids.dtype}, device: {input_ids.device}")

            out, diffusion_log = trainer.generate_for_test(
                input_ids,
            )

            diffusion_steps = [dsl.size(0) for dsl in diffusion_log]
            generated_texts = tokenizer.batch_decode(out[:, -trainer.args.max_completion_length:], skip_special_tokens=False)
            example_result = [
                {
                    "question": questions[j],
                    "prompt_input": prompts[j],
                    "generations": generated_texts[j],
                    "ground_truth": gt_answers[j],
                    "diffusion_steps": diffusion_steps[j],
                }
                for j in range(len(gt_answers))
            ]
            all_generations.extend(example_result)
            total_processed += len(generated_texts)
            wall_times.append(time.time() - start_time)

            # Print individual results
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("diffusion steps:")
            print(diffusion_steps[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")
            
            if save_processing_dir is not None:
                with open(save_processing_dir+"_processing_generations.json", "w") as f:
                    json.dump(
                        {
                            "generations": all_generations,
                            "metrics": {
                                "wall_time":sum(wall_times) / len(wall_times),
                                "total_processed": total_processed.item(),
                            },
                            "avg_diffusion_steps": 0,
                        },
                        f,
                        indent=2,
                    )

    if save_processing_dir is not None:
        # remove intermediate processing file
        os.remove(save_processing_dir+"_processing_generations.json")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


if __name__ == "__main__":
    init_seed(42)

    # Note: This evaluation script saves only model generations. A separate parser is used later to extract
    # predictions and calculate metrics.

    # parser = argparse.ArgumentParser()
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    parser.add_argument("--trainer_checkpoint_path", type=str, default=None)
    parser.add_argument("--test_model_path", type=str, default=None)
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument(
    #     "--dataset", type=str, choices=["gsm8k", "math", "countdown", "sudoku", "game24"], default="gsm8k"
    # )
    # parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--test_gen_length", type=int, default=None)
    parser.add_argument("--test_block_length", type=int, default=None)
    parser.add_argument("--test_diffusion_steps", type=int, default=None)
    parser.add_argument("--test_remasking", type=str, default=None)
    parser.add_argument("--test_prob_threshold", type=float, default=None)
    parser.add_argument("--test_dataset", type=str, choices=["gsm8k", "math", "countdown", "sudoku", "humaneval", "mbpp"], default=None)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--test_output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument("--boxed", action="store_true", default=False)
    
    # print(len(parser.parse_args_and_config(return_remaining_strings=True)))
    grpo_config, model_config, args = parser.parse_args_and_config()
    if args.test_dataset is not None:
        grpo_config.dataset = args.test_dataset
    # args.diffusion_steps = args.gen_length // 2
    num_evals = {"gsm8k": -1, "math": -1, "countdown": 256, "sudoku": 256, "humaneval": -1, "mbpp": -1}
    
    grpo_config.output_dir = args.test_output_dir
    if args.test_model_path is None and args.trainer_checkpoint_path is None:
        raise ValueError("Either test_model_path or trainer_checkpoint_path must be provided")
    
    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "countdown":
        reward_functions = [countdown_reward_func]
    elif grpo_config.dataset == "sudoku":
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
    elif grpo_config.dataset == "humaneval":
        reward_functions = [
            kodcode_reward_func,
        ]
    elif grpo_config.dataset == "mbpp":
        reward_functions = [
            kodcode_reward_func,
        ]

    tokenizer = AutoTokenizer.from_pretrained(args.test_model_path, trust_remote_code=True, padding_side="left")
    
    if args.trainer_checkpoint_path is not None:
        grpo_config = torch.load(args.trainer_checkpoint_path + "/training_args.bin", weights_only=False)
        if args.test_dataset is not None:
            grpo_config.dataset = args.test_dataset
        # 4 bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModel.from_pretrained(
            args.test_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, args.trainer_checkpoint_path)
    else:
        # Load model and tokenizer
        model = AutoModel.from_pretrained(
            args.test_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    if "dream" in args.test_model_path.lower():
        grpo_config.mask_id = model.config.mask_token_id
    grpo_config.beta=0.0
        
    model.config.use_cache = False
    trainer = DiffuGRPOTrainer(
        args=grpo_config,
        model=model,
        reward_funcs=reward_functions,
        processing_class=tokenizer if "dream" in args.test_model_path.lower() else None,
    )

    trainer.args.max_completion_length = args.test_gen_length if args.test_gen_length is not None else trainer.args.max_completion_length
    trainer.args.block_length = args.test_block_length if args.test_block_length is not None else trainer.args.block_length
    trainer.args.diffusion_steps = args.test_diffusion_steps if args.test_diffusion_steps is not None else trainer.args.diffusion_steps
    trainer.args.remasking = args.test_remasking if args.test_remasking is not None else trainer.args.remasking
    trainer.args.prob_threshold = args.test_prob_threshold if args.test_prob_threshold is not None else trainer.args.prob_threshold
    trainer.args.temperature = 0.0

    print(f"model architecture: {trainer.model}")
    print(f"config: {trainer.args}")
    print(f"Model loaded from {args.trainer_checkpoint_path}")
    print(f"Number of parameters: {sum(p.numel() for p in trainer.model.parameters())}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}")
    
    # try:
    #     args.boxed = True if "boxed" in args.trainer_checkpoint_path else args.boxed
    # except:
    #     pass

    dataset = DATASET_MAP[grpo_config.dataset](
        tokenizer,
        subsample=num_evals[grpo_config.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,  # prefill for all models
        # is_boxed=args.boxed
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
    )

    if args.trainer_checkpoint_path is not None:
        model_name = args.trainer_checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = args.test_model_path.split("/")[-1]

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    # if len(args.suffix) > 0:
    #     model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.test_output_dir, exist_ok=True)
    if args.boxed:
        filename = f"{args.test_output_dir}/{grpo_config.dataset}_{model_name}_ds{trainer.args.diffusion_steps}_bl{trainer.args.block_length}_remask{trainer.args.remasking}_pt{trainer.args.prob_threshold}_boxed_generations.json"
    else:
        filename = f"{args.test_output_dir}/{grpo_config.dataset}_{model_name}_ds{trainer.args.diffusion_steps}_bl{trainer.args.block_length}_remask{trainer.args.remasking}_pt{trainer.args.prob_threshold}_generations.json"
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping.")
        raise ValueError(f"File {filename} already exists. Skipping.")

    print(f"Saving generations to {filename}")

    metrics = evaluate(
        trainer,
        dataloader,
        save_processing_dir=f"{args.test_output_dir}/{grpo_config.dataset}_{filename.split('/')[-1].split('_generations.json')[0]}_temp{trainer.args.temperature}" if not args.dont_save else None
    )
    avg_diff_steps = np.mean([gen["diffusion_steps"] for gen in metrics["generations"]])

    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                    },
                    "trainer_config": trainer.args.to_dict(),
                    "test_model_path": args.test_model_path,
                    "trainer_checkpoint_path": args.trainer_checkpoint_path,
                    "avg_diffusion_steps": avg_diff_steps,
                },
                f,
                indent=2,
            )

