import numpy as np
from typing import List
from graph_structures import ReasoningGraph, OutcomeType, Node

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity"""
    return np.dot(emb1, emb2)

def stepwise_cosine_similarity(path_embs: List[np.ndarray], gt_embs: List[np.ndarray]) -> float:
    """Mean cosine similarity of corresponding steps"""
    min_len = min(len(path_embs), len(gt_embs))
    
    step_similarities = []
    for i in range(min_len):
        step_similarities.append(cosine_similarity(path_embs[i], gt_embs[i]))
    
    return np.mean(step_similarities)

def label_leaf_outcomes(graph: ReasoningGraph, gt_step_embs: List[np.ndarray]):
    """SIMPLE: Step-wise cosine similarity"""
    
    # STEP 1: CORRECT = mean(stepwise cos > 0.9)
    correct_nodes = []
    for node_id, node in graph.nodes.items():
        if node.is_leaf and node.outcome is None:
            step_sim = stepwise_cosine_similarity(node.step_embeddings, gt_step_embs)
            if step_sim > 0.9:
                node.outcome = OutcomeType.CORRECT
                correct_nodes.append(node)
    
    print(f"Found {len(correct_nodes)} CORRECT leaves (step-wise >0.9)")
    
    # STEP 2: NEAR_MISS vs CORRECT nodes (step-wise ≥ 0.7)
    for node_id, node in graph.nodes.items():
        if node.is_leaf and node.outcome is None:
            is_near = False
            
            # Check vs GT
            gt_step_sim = stepwise_cosine_similarity(node.step_embeddings, gt_step_embs)
            if gt_step_sim >= 0.7:
                is_near = True
            
            # Check vs correct nodes
            for correct_node in correct_nodes:
                correct_step_sim = stepwise_cosine_similarity(
                    node.step_embeddings, correct_node.step_embeddings
                )
                if correct_step_sim >= 0.7:
                    is_near = True
                    break
            
            node.outcome = (OutcomeType.NEAR_MISS if is_near 
                          else OutcomeType.FAILURE)
    
    total_leaves = len([n for n in graph.nodes.values() if n.is_leaf])
    print(f"Labeled {total_leaves} leaves total")
