import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
import random
import re
from datasets import load_dataset
from parsers import Parser, is_equiv
import torch.distributed as dist

SYSTEM_INSTRUCT = 'You are a helpful programming assistant. The user will ask you a question and you as the assistant solve it. The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively.'
SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>

Please solve the programming task below in Python. 
"""

FORMATTING_PROMT = "Note that the function declaration is {}. Your code should be wrapped in a markdown code block."
# FORMATTING_PROMT = "Your code should be wrapped in a markdown code block."

class MBPPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=SYSTEM_PROMPT,
        subsample=-1,
        is_boxed=None,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.load_test_dataset()
        # self.create_few_shot_prompt()
        

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        self.dataset = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=True)

    def create_prompt(self, input_text, function_declaration):
        # Format similar to your chat function
        # if self.num_examples > 0:
        #     prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        # else:
        #     prompt = input_text
        prompt = input_text + "\n\n" + FORMATTING_PROMT.format(function_declaration)
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCT},
            {"role": "user", "content": self.system_prompt + "\n\n" + prompt}
        ]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            return user_input + "<think>"
        else:
            return user_input

    # def load_few_shot_examples(self):
    #     if isinstance(self.dataset, MBPPDataset):
    #         train_data = load_dataset("Muennighoff/mbpp", split="test")
    #         examples = random.sample(range(len(train_data)), self.num_examples)
    #         return [train_data[example] for example in examples]
    #     else:
    #         return []

    # def create_few_shot_prompt(self):
    #     """Create few-shot prompt from dataset examples"""
    #     few_shot_examples = self.load_few_shot_examples()

    #     formatted_examples = []
    #     for example in few_shot_examples:
    #         input_text = example["prompt"]
    #         answer = example["canonical_solution"]
    #         formatted_examples.append(f"Question: {input_text}\nAnswer:\n{answer}")
    #     self.few_shot_prompt = "\n\n".join(formatted_examples)

    def __getitem__(self, idx):
        answer = self.dataset[self.subsample[idx].item()]["code"]
        for line in answer.split("\n"):
            if line.startswith("def "):
                parsed_function_name = re.findall(r"def (.*)\(", line)[0]
                function_declaration = line
                break
        question = self.dataset[self.subsample[idx].item()]["prompt"]
        # answer = self.dataset[self.subsample[idx].item()]["canonical_solution"]
        test_imports = "\n".join(self.dataset[self.subsample[idx].item()]["test_imports"])
        test_code = "\n    ".join(["def test_test():"]+self.dataset[self.subsample[idx].item()]["test_list"])
        test_code = f"from solution import *\nfrom solution import {parsed_function_name}\n\n{test_imports}\n" + test_code + "\n"

        prompt = self.create_prompt(question, function_declaration)
        
        return prompt, question, answer, test_code

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        test_codes = [item[3] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": test_codes, "prompts": prompts ,"test_codes": test_codes}
