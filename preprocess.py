import re
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

def split_into_steps(text: str) -> List[str]:
    """Split response into reasoning steps"""
    text = text.replace("\n", " ")
    steps = re.split(r'\s*(?:\d+\.\s+|(?<=\.)\s+)', text)
    steps = [s.strip() for s in steps if len(s.strip()) > 3]
    return steps

def encode_steps(encoder: SentenceTransformer, steps: List[str]) -> np.ndarray:
    """
    Encode list of step texts into embeddings
    
    Input:  ["Step 1: 290 bananas", "Step 2: divide by 2"]
    Output: np.array shape (2, 384) = [[0.1,-0.2,...], [0.3,0.1,...]]
    """
    if not steps:
        return np.array([])
    
    embeddings = encoder.encode(
        steps,
        convert_to_numpy=True,
        normalize_embeddings=True, 
        show_progress_bar=False
    )
    
    return embeddings  # Shape: (num_steps, embedding_dim)
