from typing import List, Dict,Tuple,Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch 

from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_lm(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )

    # Padding fix for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Base kwargs
    kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

        # Try FlashAttention-2 first
        try:
            kwargs["attn_implementation"] = "flash_attention_2"
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            print("✓ FlashAttention-2 enabled")
        except Exception as e:
            # Fallback to PyTorch SDPA / standard attention
            print(f"⚠ FlashAttention unavailable, falling back ({e})")
            kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    model.eval()
    return model, tokenizer


def load_encoder(encoder_name: str) -> SentenceTransformer:
    """Load ANY SentenceTransformer"""
    return SentenceTransformer(encoder_name, device="cuda" if torch.cuda.is_available() else "cpu")
