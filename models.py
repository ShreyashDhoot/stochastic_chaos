from typing import List, Dict,Tuple,Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch 

def load_lm(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Pad token fix
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Robust model loading
    kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": "mixtral" in model_name.lower(),
    }
    
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    
    # Warmup (optional, speeds up first inference)
    model.eval()
    
    return model, tokenizer

def load_encoder(encoder_name: str) -> SentenceTransformer:
    """Load ANY SentenceTransformer"""
    return SentenceTransformer(encoder_name, device="cuda" if torch.cuda.is_available() else "cpu")
