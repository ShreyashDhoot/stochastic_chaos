from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from graph_structure import ReasoningGraph,Node, OutcomeType,benchmarking_config,QuestionMetrics,ModelMetrics,BenchmarkResults

def greedy_support_ratio(graph: ReasoningGraph) -> float:
    """greedy_logprob / max(non_greedy_logprobs)"""
    path_logprobs = graph.path_logprobs  # {"greedy": -2.3, "sample_0": -3.1, "sample_1": -2.8}
    
    if "greedy" not in path_logprobs:
        return 0.0
    
    greedy_logprob = path_logprobs["greedy"]
    
    # Get ALL non-greedy logprobs
    non_greedy_logprobs = [lp for tag, lp in path_logprobs.items() if tag != "greedy" and tag.startswith("sample_")]
    
    if not non_greedy_logprobs:
        return 1.0  # No competition = perfect support
    
    max_non_greedy = max(non_greedy_logprobs)
    
    # Edge case: greedy is worse than all samples
    if greedy_logprob < max_non_greedy:
        return 0.0
    
    return float(greedy_logprob / max_non_greedy)  # [0,1] range

def path_entropy(graph)->float:
    """COMPUTES Shannon entropy H(p)"""
    logprobs = np.array(list(graph.path_logprobs.values()))
    probs = softmax(logprobs)
    return float(-np.sum(probs * np.log2(probs + 1e-8)))

def collapse_failure_rate(question_metrics_list: List[QuestionMetrics]) -> float:
    """% of greedy failures where samples succeed (true collapse!)"""
    greedy_wrong_questions = [
        qm for qm in question_metrics_list 
        if qm.greedy_outcome != OutcomeType.CORRECT
    ]
    
    if not greedy_wrong_questions:
        return 0.0  # No greedy failures = no collapse
    
    # Collapse = greedy wrong BUT has correct leaves in samples
    collapse_cases = [
        qm for qm in greedy_wrong_questions 
        if qm.num_correct_leaves > 0  # Samples found correct answer!
    ]
    
    return len(collapse_cases) / len(greedy_wrong_questions)


def exploration_gain(saved_graph_paths: List[Path]) -> float:
    """
    # Questions where MULTI-SAMPLING succeeds - # where GREEDY succeeds
    = multi_correct - greedy_correct
    """
    greedy_correct = 0
    multi_correct = 0
    
    for graph_path in saved_graph_paths:
        with open(graph_path, "rb") as f:
            graph: ReasoningGraph = pickle.load(f)
        
        # Greedy correct?
        greedy_leaf = graph.nodes[graph.greedy_path[-1]]
        if greedy_leaf.outcome == OutcomeType.CORRECT:
            greedy_correct += 1
        
        # Multi-sampling correct? 
        sample_leaves = [n for n in graph.nodes.values() 
                        if n.is_leaf and not n.is_greedy]
        if any(leaf.outcome == OutcomeType.CORRECT for leaf in sample_leaves):
            multi_correct += 1
    
    return multi_correct - greedy_correct 


def extract_question_metrics(graph: ReasoningGraph, question: Dict,question_id: int) -> QuestionMetrics:
    """Pure extraction - no graph storage!"""
    leaves = [n for n in graph.nodes.values() if n.is_leaf]
    correct_leaves = [n for n in leaves if n.outcome == OutcomeType.CORRECT]
    
    return QuestionMetrics(
        question_id=question_id,
        path_diversity=len(graph.path_logprobs) > 1,
        greedy_support=greedy_support_ratio(graph),
        path_entropy=path_entropy(graph),
        greedy_outcome=graph.greedy_path[-1].outcome if graph.greedy_path else OutcomeType.FAILURE,
        num_correct_leaves=len(correct_leaves)
    )