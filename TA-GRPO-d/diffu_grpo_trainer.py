import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import wandb
import json
import os

import sys
import io
from contextlib import redirect_stdout

#garbage collection
import gc

if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


#for dream model generation
#code copy from https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/tree/main
import torch.distributions as dists
def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0

class DiffuGRPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        os.makedirs(f"{self.args.output_dir}/logs", exist_ok=True)

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        mask_seeds = inputs["mask_seeds"]

        # Combine prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # only compute logits for completion tokens

        # Get the current iteration index and corresponding mask seed
        this_itr_idx = self._step % self.args.num_iterations
        this_itr_mask_seed = mask_seeds[this_itr_idx]
        input_ids = input_ids.unsqueeze(0)
        per_token_logps = self._get_per_token_logps(model, input_ids, logits_to_keep, [this_itr_mask_seed])
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"][this_itr_idx].squeeze(0)
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = (
            inputs["old_per_token_logps"][this_itr_idx].squeeze(0)
            if self.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        
        
        if len(advantages.shape) == 1:
            advantages = advantages.unsqueeze(1)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        
        # per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        # per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise
    
    def generate_for_test(
        self,
        prompt,
        steps=None,
        gen_length=None,
        block_length=None,
        temperature=None,
        cfg_scale=None,
        remasking=None,
        mask_id=None,
        prob_threshold=None,
        prob_delta=None,
        prob_min_gen=None,
    ):
        
        gen_length = self.args.max_completion_length if gen_length is None else gen_length
        block_length = self.args.block_length if block_length is None else block_length
        steps = self.args.diffusion_steps if steps is None else steps
        temperature = 0.0
        cfg_scale = 0.0
        mask_id = self.args.mask_id if mask_id is None else mask_id
        prob_threshold = self.args.prob_threshold if prob_threshold is None else prob_threshold
        prob_delta = self.args.prob_delta if prob_delta is None else prob_delta
        prob_min_gen = self.args.prob_min_gen if prob_min_gen is None else prob_min_gen
        remasking = self.args.remasking if remasking is None else remasking
        
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            return self.generate(
                model=unwrapped_model,
                prompt=prompt,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking="probabilistic_test" if remasking=="probabilistic" else remasking,
                mask_id=mask_id,
                prob_threshold=prob_threshold,
                prob_delta=prob_delta,
                prob_min_gen=prob_min_gen,
                return_diffusion_log=True,
            )

    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        prob_threshold=0.7,
        prob_delta=0.5,
        prob_min_gen=2,
        mask_id=126336,
        return_diffusion_log=False,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        """we combine generation code from Dream and llada for more general usage (dream code referenced by https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/tree/main)"""
        
        with torch.cuda.amp.autocast(enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            if return_diffusion_log:
                diffusion_seq_log = [[] for _ in range(bs)]
                
            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)
            if remasking == "probabilistic" or remasking == "probabilistic_test":
                original_steps_per_block = steps_per_block
                steps_per_block = block_length // prob_min_gen

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    if (x[:, start_idx:end_idx] == mask_id).sum() == 0:
                        print(f"early stop at block {num_block}, step {i} original step {original_steps_per_block} on {x.shape[0]} samples")
                        break

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if "dream" in self.args.model_path.lower():
                                if cfg_scale > 0.0:
                                    un_x = x.clone()
                                    un_x[prompt_index] = mask_id
                                    x_ = torch.cat([x, un_x], dim=0)

                                    # Get logits in a single forward pass
                                    logits = model(x_,"full",None).logits
                                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                                else:
                                    logits = model(x,"full",None).logits
                                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
                                
                                if remasking == "random":
                                    _, x0 = sample_tokens(logits, temperature=temperature, top_p=0.95)
                                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                                elif remasking == "low_confidence" or remasking == "probabilistic" or remasking == "probabilistic_test":
                                    # x0_p, x0 = sample_tokens(logits, temperature=0.7, top_p=0.97)
                                    _, x0 = sample_tokens(logits, temperature=temperature, top_p=0.95)
                                    p = F.softmax(logits.to(dtype), dim=-1)
                                    x0_p = torch.squeeze(
                                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                    )
                                else:
                                    raise NotImplementedError(remasking)
                            else:
                                if cfg_scale > 0.0:
                                    un_x = x.clone()
                                    un_x[prompt_index] = mask_id
                                    x_ = torch.cat([x, un_x], dim=0)

                                    # Get logits in a single forward pass
                                    logits = model(x_).logits
                                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                                else:
                                    logits = model(x).logits

                                # Apply Gumbel noise for sampling
                                _, x0 = sample_tokens(logits, temperature=temperature, top_p=0.95)

                                # Handle remasking strategy
                                if remasking == "random":
                                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                                elif remasking == "low_confidence" or remasking == "probabilistic" or remasking == "probabilistic_test":
                                    p = F.softmax(logits.to(dtype), dim=-1)
                                    x0_p = torch.squeeze(
                                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                    )
                                else:
                                    raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            remask_flag = []
                            if remasking == "probabilistic":
                                tmp_confidence = confidence.clone()
                                for j in range(confidence.shape[0]):
                                    if (x[j, start_idx:end_idx] == mask_id).sum() == 0:
                                        remask_flag.append(False)
                                        continue
                                    remask_flag.append(True)
                                    
                                    confidence[j] = (confidence[j] - prob_threshold).clamp(min=0) / (1 - prob_threshold)
                                    confidence[j] = confidence[j] ** prob_delta
                                    
                                    transfer_index[j] = confidence[j] > torch.rand_like(confidence[j]) #probabilistic denoising based on confidence
                                    
                                    if transfer_index[j].sum() < prob_min_gen:
                                        _, additional_indices = torch.topk(tmp_confidence[j], k=prob_min_gen-transfer_index[j].sum().item())
                                        transfer_index[j, additional_indices] = True
                                        
                            elif remasking == "probabilistic_test":
                                tmp_confidence = confidence.clone()
                                for j in range(confidence.shape[0]):
                                    if (x[j, start_idx:end_idx] == mask_id).sum() == 0:
                                        remask_flag.append(False)
                                        continue
                                    remask_flag.append(True)
                                    transfer_index[j] = confidence[j] > prob_threshold
                                    
                                    if transfer_index[j].sum() < prob_min_gen:
                                        _, additional_indices = torch.topk(tmp_confidence[j], k=prob_min_gen-transfer_index[j].sum().item())
                                        transfer_index[j, additional_indices] = True
                            else:
                                for j in range(confidence.shape[0]):
                                    num_tokens = num_transfer_tokens[j, i].item()
                                    if num_tokens > 0:
                                        remask_flag.append(True)
                                        _, select_index = torch.topk(confidence[j], k=num_tokens)
                                        transfer_index[j, select_index] = True
                                        continue
                                    remask_flag.append(False)

                            x[transfer_index] = x0[transfer_index]
                            
                            if return_diffusion_log:
                                for rm_f_i, rm_f in enumerate(remask_flag):
                                    if rm_f:
                                        diffusion_seq_log[rm_f_i].append(x[rm_f_i].detach().cpu().clone())
                                    else:
                                        pass
                            
                            del x0, confidence, transfer_index
            
            if return_diffusion_log:
                for i in range(bs):
                    diffusion_seq_log[i] = torch.stack(diffusion_seq_log[i], dim=0) if len(diffusion_seq_log[i]) > 0 else torch.empty((0, x.shape[1]), dtype=torch.long)
                return x, diffusion_seq_log

            return x

    def forward_process(self, batch, prompt_index, mask_id, seed=None):
        set_seed(seed)
        b, l = batch.shape
        t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt

        # Create a random matrix to decide whether each prompt token is masked
        random_matrix = torch.rand((b, l), device=batch.device)

        # For prompt tokens: mask if random_matrix < t_p
        # For completion tokens: always mask
        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index  # all completion tokens are masked
        is_mask = is_mask_prompt | is_mask_completion

        # Create a noisy (masked) batch
        noisy_batch = torch.where(is_mask, mask_id, batch)

        # Build p_mask, the probability that each token is masked under this scheme
        #   - p_mask[i, j] = t_p[i] if it's a prompt token
        #   - p_mask[i, j] = 1      if it's a completion token
        p_mask = torch.where(
            prompt_index,
            t_p.unsqueeze(1),  # prompt token probability
            torch.ones_like(t_p).unsqueeze(1),  # completion token probability
        )

        return noisy_batch, p_mask

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])

        input = batch
        logits = model(input).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)

    def _get_per_token_logps(self, model, input_ids, logits_to_keep, mask_seeds):
        """
        Calculate per-token log probabilities.
        """
        num_iterations, batch_size, seq_len = input_ids.size()
        device = input_ids.device
        per_token_logps = torch.zeros(num_iterations, batch_size, logits_to_keep, device=device)

        # Verify mask_seeds length: one seed per iteration
        assert (
            len(mask_seeds) == num_iterations
        ), f"Expected mask_seeds length to be {num_iterations}, got {len(mask_seeds)}"

        prompt_length = seq_len - logits_to_keep
        prompt_index = torch.zeros(seq_len, dtype=torch.bool, device=device)
        prompt_index[:prompt_length] = True  # Mark prompt tokens as True

        # applying masks
        all_perturbed_seqs = []
        all_expanded_inputs = []
        for iter_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iter_idx]  # [batch_size, seq_len]
            perturbed_seq, _ = self.forward_process(
                expanded_input, prompt_index, self.args.mask_id, seed=mask_seed
            )
            all_perturbed_seqs.append(perturbed_seq)
            all_expanded_inputs.append(expanded_input)

        # Concatenate all iterations into a single batch
        perturbed_seq = torch.cat(all_perturbed_seqs, dim=0)  # [num_iterations * batch_size, seq_len]
        expanded_input = torch.cat(all_expanded_inputs, dim=0)  # [num_iterations * batch_size, seq_len]

        # Get model predictions for the combined batch
        logits = self.get_logits(
            model, perturbed_seq, prompt_index, self.args.cfg_scale, self.args.mask_id
        )  # [num_iterations * batch_size, seq_len, vocab_size]

        # Calculate cross-entropy loss for completion tokens only
        if "dream" in self.args.model_path.lower():
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)  # shift for dream model
        completion_logits = logits[
            :, -logits_to_keep:, :
        ]  # [num_iterations * batch_size, logits_to_keep, vocab_size]
        completion_targets = expanded_input[
            :, -logits_to_keep:
        ]  # [num_iterations * batch_size, logits_to_keep]
        flat_logits = completion_logits.reshape(-1, completion_logits.size(-1))
        flat_targets = completion_targets.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Convert to log probabilities and reshape
        completion_log_probs = -loss.view(num_iterations * batch_size, logits_to_keep)
        per_token_logps = completion_log_probs.view(num_iterations, batch_size, logits_to_keep)

        # Clean up memory
        del perturbed_seq, logits, all_perturbed_seqs, all_expanded_inputs
        torch.cuda.empty_cache()
        per_token_logps = per_token_logps.to(torch.float32)
        return per_token_logps

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        with torch.no_grad():
            if mode == "train":
                if self.state.global_step % self.num_iterations == 0:
                    inputs = self._generate_and_score_completions(inputs)
                    self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
                else:
                    inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
                self._step += 1
            else:
                # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
                inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        
        save_dict = {}

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        save_dict["prompts"] = prompts

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            prompt_completion_ids_all = []
            diffusion_seq_log_all = []
            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]
                # WARNING: Attention masks are not currently used during generation.
                # This works fine as we set num_generations == per_device_train_batch_size (no padding tokens created) in our config, but may cause
                # unintended attention to padding tokens when num_generations is smaller.
                # As currently we find Llada's modeling file does not handle attention mask. We will address this in future update soon.
                batch_prompt_completion_ids, diffusion_seq_log = self.generate(
                    model=unwrapped_model,
                    prompt=batch_prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                    prob_threshold=self.args.prob_threshold,
                    prob_delta=self.args.prob_delta,
                    prob_min_gen=self.args.prob_min_gen,
                    return_diffusion_log=True,
                )
                prompt_completion_ids_all.append(batch_prompt_completion_ids)
                diffusion_seq_log_all.extend(diffusion_seq_log)

                del batch_prompt_ids, batch_prompt_mask, batch_prompt_completion_ids
                torch.cuda.empty_cache()

            prompt_completion_ids = torch.cat(prompt_completion_ids_all, dim=0)

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        if self.args.random_masking:
            # use random seeds for every iterations in GRPO iterations
            mask_seeds = torch.randint(0, 2**12, (self.num_iterations,), device=device)
        else:
            # use fixed seeds for every iterations in GRPO iterations
            mask_seeds = [42] * self.num_iterations

        all_old_per_token_logps = []
        all_ref_per_token_logps = []
        
        # GPU memory issue we cannot compute all iteration in one batch, so we compute them sequentially.
        with torch.no_grad():
            if self.num_iterations > 1:
                for i in range(self.num_iterations):
                    prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0)
                    old_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids_expanded, logits_to_keep, [mask_seeds[i]]
                    )
                    all_old_per_token_logps.append(old_per_token_logps)
                all_old_per_token_logps = torch.cat(all_old_per_token_logps, dim=0)
            else:
                prompt_completion_ids_expanded = prompt_completion_ids.unsqueeze(0)
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    for i in range(self.num_iterations):
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids_expanded, logits_to_keep, [mask_seeds[i]]
                        )
                        all_ref_per_token_logps.append(ref_per_token_logps)
                all_ref_per_token_logps = torch.cat(all_ref_per_token_logps, dim=0)

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        save_dict["completions"] = completions
        
        def get_rewards(prom, comp, skip_flag=None):
            if skip_flag is not None:
                rewards_per_func = torch.zeros(len(prom) - sum(skip_flag), len(self.reward_funcs), device=device)
            else:
                rewards_per_func = torch.zeros(len(prom), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                else:
                    reward_func_name = reward_func.__name__
                with profiling_context(self, reward_func_name):

                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    if reward_func_name == "coding_reward_func":
                        reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                    
                    if skip_flag is not None:
                        prom_tmp = [p for p, s in zip(prom, skip_flag) if not s]
                        comp_tmp = [c for c, s in zip(comp, skip_flag) if not s]
                    else:
                        prom_tmp = prom
                        comp_tmp = comp
                        
                    if len(prom_tmp) == 0:
                        continue
                    
                    output_reward_func = reward_func(
                        prompts=prom_tmp,
                        completions=comp_tmp,
                        step=self._step,
                        run_name=self.args.output_dir,
                        **reward_kwargs,
                    )
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            # If all reward functions return None for a given row, issue a detailed warning
            if torch.isnan(rewards_per_func).all(dim=1).any():
                nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
                row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
                row_reward_kwargs["prompt"] = prom_tmp[nan_row_idx]
                row_reward_kwargs["completion"] = comp_tmp[nan_row_idx]
                warnings.warn(
                    f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                    "Please ensure that at least one reward function returns a valid reward."
                )

            rewards_per_func = gather(rewards_per_func)
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

            return rewards, rewards_per_func

        rewards, rewards_per_func = get_rewards(prompts, completions)
        save_dict["rewards"] = rewards.cpu().numpy().tolist()
        
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        # Count prompts with zero std deviation
        zero_std_count = (std_grouped_rewards < 1e-6).sum().item()  # Using a small threshold
        total_prompts = std_grouped_rewards.size(0)
        zero_std_ratio = zero_std_count / total_prompts if total_prompts > 0 else 0.0

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        save_dict["advantages"] = advantages.cpu().numpy().tolist()

        if self.args.use_trajectory_aware_rewards:
            # CALCULATE TRAJECTORY-AWARE REWARDS
            all_dsl_rewards = torch.ones([advantages.shape[0], completion_ids.shape[1]], device=advantages.device)
            all_completion_text_decoded = []
            all_mask_prob_per_token = []
            for dsl_i, dsl in enumerate(diffusion_seq_log_all):
                dsl_completions_text = self.processing_class.batch_decode(dsl[:, prompt_length:], skip_special_tokens=True)
                all_completion_text_decoded.append(dsl_completions_text)
                
                diffusion_history = (dsl != self.args.mask_id) # [num_steps, seq_len] boolean tensor indicating whether each token is masked or not at each diffusion step
                first_diffusion_step = torch.argmax(diffusion_history.int(), dim=0) #argmax will return the first occurrence of the maximum value, which in this case is the first time the token is not masked anymore, i.e., the first diffusion step where the token is generated.
                
                mask_ratio_per_step = (dsl == self.args.mask_id).int().sum(dim=1) / gen_length # calcuate the ratio of masked tokens at each diffusion step earlier diffusion steps will have higher mask ratio, later diffusion steps will have lower mask ratio
                
                mask_prob_per_token = mask_ratio_per_step[first_diffusion_step] # assign the mask ratio of the first diffusion step, earlier dinoised tokens will have higher mask ratio, later denoised tokens will have lower mask ratio
                mask_prob_per_token = mask_prob_per_token[-logits_to_keep:]
                all_mask_prob_per_token.append(mask_prob_per_token) # mask ratio score of all completion tokens
            
            save_dict["diffusion_seq_log"] = all_completion_text_decoded  
                
            all_mask_prob_per_token = torch.stack(all_mask_prob_per_token, dim=0) # [bs, seq_len(completion)]
            
            max_completion_len = max([len(x) for x in all_completion_text_decoded])
            all_dsl_rewards = []
            prev_dsl_completion_text = [""] * len(all_completion_text_decoded)
            prev_dsl_rewards = torch.zeros(len(all_completion_text_decoded), device=device)
            for max_i in range(max_completion_len):
                dsl_completions_text = []
                skip_flag = torch.tensor([False] * len(all_completion_text_decoded), device=device)
                for i in range(len(all_completion_text_decoded)):
                    if max_i < len(all_completion_text_decoded[i]):
                        dsl_completions_text.append(all_completion_text_decoded[i][max_i])
                    else:
                        dsl_completions_text.append(all_completion_text_decoded[i][-1]) # padding last result if some sample's diffusion step is shorter
            
                    if dsl_completions_text[i] == prev_dsl_completion_text[i]:
                        skip_flag[i] = True
                
                if is_conversational(inputs[0]):
                    dsl_completions = []
                    for prompt, completion in zip([prompts[dsl_i]]*len(dsl_completions_text), dsl_completions_text):
                        bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                        dsl_completions.append([{"role": "assistant", "content": bootstrap + completion}])
                else:
                    dsl_completions = dsl_completions_text
                
                with io.StringIO() as buf, redirect_stdout(buf): #for skip print
                    # skip reward calculation if the completion is same as previous step.
                    dsl_rewards, _ = get_rewards(prompts, dsl_completions, skip_flag=skip_flag) # only retrun non skip flags rewards
                    
                tmp_reward = prev_dsl_rewards.clone()
                tmp_reward[~skip_flag] = dsl_rewards
                dsl_rewards = tmp_reward
                
                all_dsl_rewards.append(dsl_rewards.unsqueeze(1))
                
                prev_dsl_completion_text = dsl_completions_text
                prev_dsl_rewards = dsl_rewards
            
            all_dsl_rewards = torch.cat(all_dsl_rewards, dim=1)
            save_dict["all_dsl_rewards"] = all_dsl_rewards.cpu().numpy().tolist()
            

            cand_dsl_rewards = all_dsl_rewards.sum(dim=-1)
            save_dict["cand_dsl_rewards"] = cand_dsl_rewards.cpu().numpy().tolist()
            # mean and std
            # m = cand_dsl_rewards.mean()
            # s = cand_dsl_rewards.std()
            # advan_scale = (cand_dsl_rewards - m) / (s + 1e-8)
            n = self.num_generations
            cand_d = cand_dsl_rewards.view(-1, n)            # [num_prompts, n]
            mu = cand_d.mean(dim=1, keepdim=True)            # [num_prompts, 1]
            sd = cand_d.std(dim=1, keepdim=True)             # [num_prompts, 1]
            z  = (cand_d - mu) / (sd + 1e-8)                 # [num_prompts, n]
            advan_scale = z.view(-1)                         # [num_prompts*n]
            
            advan_scale_per_token = advan_scale.unsqueeze(1) * all_mask_prob_per_token.to(advantages.device)
            advantages = advantages.unsqueeze(1) + advan_scale_per_token * advantages.unsqueeze(1) * self.args.delta_scale

            save_dict["advan_scale"] = advan_scale.cpu().numpy().tolist()
            save_dict["scaled_advantages_per_token"] = advantages.cpu().numpy().tolist()
            save_dict["diffusion_seq_lengths"] = [int(dsl.size(0)) for dsl in diffusion_seq_log_all]
            json.dump(save_dict, open(f"{self.args.output_dir}/logs/step_{self.state.global_step}_generation_log.json", "w"), indent=4)
        
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["avg_diffusion_step"].append(sum([dsl.size(0) for dsl in diffusion_seq_log_all])/len(diffusion_seq_log_all) if len(diffusion_seq_log_all) > 0 else 0)

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                        "num_diffusion_step": [dsl.size(0) for dsl in diffusion_seq_log_all],
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": all_old_per_token_logps,
            "ref_per_token_logps": all_ref_per_token_logps,
            "advantages": advantages,
            "mask_seeds": mask_seeds,  # Store all mask seeds for consistent mask patterns
        }
