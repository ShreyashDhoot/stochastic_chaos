import torch 
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import List, Dict,Tuple,Optional, Union

#calculate log probs 
def apply_top_p_nucleus(probs: torch.Tensor, top_p: float, min_tokens: int = 1) -> torch.Tensor:
    """
    Apply top-p (nucleus) truncation and renormalization.
    Returns a NEW tensor (no in-place mutation).
    """
    # Sort probabilities (descending)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Cumulative probability
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Determine which sorted positions to remove
    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., :min_tokens] = False  # always keep at least min_tokens

    # Map sorted mask back to vocab order
    remove_mask = torch.zeros_like(probs, dtype=torch.bool)
    remove_mask[sorted_indices] = sorted_remove

    # Apply mask
    filtered_probs = probs.clone()
    filtered_probs[remove_mask] = 0.0

    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    return filtered_probs


def compute_chain_logprob(model,tokenizer,prompt: str,chain_text: str,temperature: float = 1.0,top_p: float = 1.0,) -> float:
    """
    Compute log p(chain | prompt) under the EXACT decoding distribution:
    - temperature scaling
    - top-p nucleus truncation + renormalization
    This matches the definition used for greedy-support ratio.
    """
    full_text = prompt + chain_text
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

    prompt_len = len(tokenizer(prompt)["input_ids"])
    chain_token_ids = input_ids[0, prompt_len:]

    log_prob = 0.0
    current_input = input_ids[:, :prompt_len].clone()

    for next_token_id in chain_token_ids:
        with torch.no_grad():
            outputs = model(current_input)
            logits = outputs.logits[0, -1, :]

        # Temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Softmax
        probs = torch.softmax(logits, dim=-1)

        # Top-p nucleus
        if top_p < 1.0:
            probs = apply_top_p_nucleus(probs, top_p)

        # Probability of the observed token
        token_prob = probs[next_token_id].item()
        log_prob += np.log(token_prob + 1e-8)  # epsilon smoothing

        # Append token
        current_input = torch.cat(
            [current_input, next_token_id.view(1, 1)], dim=1
        )

    avg_path_logprobs=log_prob/max(1, len(chain_token_ids))
    return avg_path_logprobs


#generate answer for a question using greedy decoding i.e T=0
def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> Tuple[str, float]:
    """
    Generate deterministically (T=0) AND compute path log-probability
    Returns: (generated_text: str, logprob: float)
    """
    messages = [{"role": "system", "content": "reason out the logic in steps and give the final answer"},
                {"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    model_inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            #no sampling 
            do_sample=False, 
            temperature=1.0,      
            top_p=1.0,               
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    #extract answer 
    prompt_len = model_inputs.input_ids.shape[1]
    # Remove prompt tokens
    generated_ids = outputs[0, prompt_len:] 
    greedy_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    #compute autoregressive log probs 
    logprob = compute_chain_logprob(
        model, tokenizer, full_prompt, greedy_text, 
        temperature=1.0, top_p=1.0  # Raw model distribution for greedy
    )
    
    return greedy_text, logprob

##generate answers for multiple decoding 
def generate_multi_sample(model, tokenizer, prompt: str, k: int, 
                         temperature: float, top_p: float, max_new_tokens: int):
    """
    Generate K stochastic samples + their logprobs
    Returns: K generated answers: List[str], Autoregressive Log probs for each answer :List[float]
    """
    
    #build prompt
    messages = [
        {"role": "system", "content": "reason out the logic in steps and give the final answer"},
        {"role": "user", "content": prompt}
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    #convert prompt to token 
    model_inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    prompt_len = model_inputs.input_ids.shape[1] 
    
    #generate text
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    
    texts = tokenizer.batch_decode(
        outputs[:, prompt_len:],  # Shape: [k, seq_len]
        skip_special_tokens=True
    ) 
    
    #compute log probs for all k text 
    logprobs = []
    for text in texts: 
        logprob = compute_chain_logprob(
            model, tokenizer, full_prompt, text,
            temperature=temperature, top_p=top_p 
        )
        logprobs.append(logprob)
    print(f'Generated {k} samples with logprobs:{logprobs}')
    return texts, logprobs 
