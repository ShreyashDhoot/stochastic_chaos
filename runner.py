#!/usr/bin/env python3
import argparse
import datasets
import re
import pandas as pd
import torch
import numpy as np 
import pickle
from enum import Enum
from typing import List, Dict,Tuple,Optional, Union
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from graph_structure import ReasoningGraph,Node, OutcomeType
from models import load_lm, load_encoder
from generation import generate_greedy, generate_multi_sample
from preprocess import split_into_steps, encode_steps
from dag_build import add_chain_to_graph
from outcome import label_leaf_outcomes
from metrics import (
    has_path_diversity, 
    greedy_support_ratio, 
    collapse_failure_rate,
    extract_question_metrics,
    exploration_gain
)
from answer_parser import extract_final_answer, answers_match, normalize_answer

@dataclass
class benchmarking_config:
    model_name:str
    encoder_name:str="all-MiniLM-L6-v2"
    dataset_name: str
    num_samples: int = 8
    temperature: float = 0.7
    top_p: float = 0.9
    similarity_threshold: float = 0.9
    nearmiss_threshold: float = 0.7
    max_new_tokens: int = 1024

@dataclass
class QuestionMetrics:
    question_id: int
    path_diversity: bool
    greedy_support: float
    path_entropy: float
    greedy_outcome: OutcomeType
    num_correct_leaves: int
    
@dataclass  
class ModelMetrics:
    model_name: str
    dataset_name: str
    collapse_failure: float      # Model+Dataset level
    exploration_gain: float      # Model+Dataset level
    avg_path_entropy: float      # Model+Dataset level
    question_metrics: List[QuestionMetrics]  # For plotting!

@dataclass
class BenchmarkResults:
    model_metrics: ModelMetrics
    # No graphs - lightweight!
    
    def to_question_df(self) -> pd.DataFrame:
        """Flat DataFrame for per-question plots"""
        return pd.DataFrame([qm.__dict__ for qm in self.model_metrics.question_metrics])

#function to run the whole pipeline per question
def run_for_question(model,tokenizer,encoder:SentenceTransformer,config:benchmarking_config,question=Dict)->ReasoningGraph:
    question_text=question['question']
    ground_truth_answer=question['answer']
    ground_truth_steps=question.get('solution_steps',[])

    print(f"preprocessing question : {question_text[:60]}...")

    #generating the greedy response 
    greedy_answer,greedy_logprobs=generate_greedy(model,tokenizer,question_text,config.max_new_tokens)
    ##generate response for multiple sample 
    sample_texts, sample_logprobs = generate_multi_sample(model, tokenizer, question_text, config.num_samples,config.temperature, config.top_p, config.max_new_tokens)

    ##creating reasoning graph 
    graph = ReasoningGraph(question_id=question.get('id', 0),question_text=question_text,ground_truth_answer=ground_truth_answer,similarity_threshold=config.similarity_threshold,nearmiss_overlap_threshold=config.nearmiss_threshold)

    #build reasoning graph for greedy sampling 
    greedy_steps=split_into_steps(greedy_answer)
    greedy_embs = encode_steps(encoder, greedy_steps)
    add_chain_to_graph(graph, greedy_steps, greedy_embs,is_greedy=True, sample_tag="greedy")
    graph.path_logprobs["greedy"] = greedy_logprobs

    #build reasoning graph for multiplesampling 
    for i,(text,logprob) in enumerate(zip(sample_texts,sample_logprobs)):
        steps=split_into_steps(text)
        if not steps:
            continue
        embs=encode_steps(encoder,steps)
        add_chain_to_graph(graph,steps,embs,is_greedy=False,sample_tag=f"sample_{i}")
        graph.path_logprobs[f"sample_{i}"]=logprob

    #breaking in steps and encoding the ground truth answer and seperating 
    #this needs to be list of np.ndarray come back while writing the main 
    gt_step_embs = encode_steps(encoder, ground_truth_steps) if ground_truth_steps else None
    
    #labelling the leaf outcomes 
    label_leaf_outcomes(graph,gt_step_embs)

    print(graph)
    return graph

def run_benchmark(dataset: List[Dict], config:benchmarking_config, output_dir: str = "results") -> list[QuestionMetrics]:
    """Run complete benchmark"""
    output_path = Path(output_dir) / f"{config.dataset_name}_{config.model_name.split('/')[-1]}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {config.model_name}")
    model, tokenizer = load_lm(config.model_name)
    
    print(f"Loading encoder: {config.encoder_name}")
    encoder = load_encoder(config.encoder_name)

    question_metrics_list=[]

    for i, question in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}]")
        graph = run_for_question(model, tokenizer, encoder, question, config)
        q_metrics=extract_question_metrics(graph,question,i)
        question_metrics_list.append(q_metrics)
        # Save per-question graph
        with open(output_path / f"q_{i}.pkl", "wb") as f:
            pickle.dump(graph, f)
    
    # Save question-level CSV (plot-ready!)
    question_df = pd.DataFrame([qm.__dict__ for qm in question_metrics_list])
    question_df.to_csv(output_path / "question_metrics.csv", index=False)

    with open(output_path / "question_metrics.pkl", "wb") as f:
        pickle.dump(question_metrics_list, f)
    
    print(f"Saved {len(question_metrics_list)} question metrics + graphs")
    return question_metrics_list

def main(model_name: str, hf_dataset: str, output_dir: str = "results"):
    """Load existing results → Model/Dataset metrics → ALL plots!"""

    # 1. Load QuestionMetrics (no graphs needed!)
    output_path = Path(output_dir) / f"{hf_dataset}_{model_name.split('/')[-1]}"
    with open(output_path / "question_metrics.pkl", "rb") as f:
        question_metrics_list = pickle.load(f)
    
    question_df = pd.DataFrame([qm.__dict__ for qm in question_metrics_list])
    
    # 2. Calculate Model+Dataset metrics
    model_metrics = ModelMetrics(
        model_name=model_name,
        dataset_name=hf_dataset,
        num_questions=len(question_metrics_list),
        collapse_failure=collapse_failure_rate(question_metrics_list),
        exploration_gain=exploration_gain(question_metrics_list),
        avg_path_entropy=np.mean(question_df.path_entropy),
    )
    
    # 3. Save ModelMetrics
    with open(output_path / "model_metrics.pkl", "wb") as f:
        pickle.dump(model_metrics, f)
    
    print(f" Complete analysis for {model_name} on {hf_dataset}")
    return model_metrics

def load_hf_dataset(dataset_name: str, split: str = "test", num_samples: int = 1000) -> List[Dict]:
    """Load HF dataset → List[Dict]"""
    print(f"Loading {dataset_name} ({split}[:{num_samples}])")
    ds = datasets.load_dataset(dataset_name, split=split[:num_samples])
    
    # Standardize format (GSM8K, etc.)
    return [{
        'id': i,
        'question': row['question'],
        'answer': row.get('answer', row.get('final_answer', '')),
        'solution_steps': row.get('solution', [])  # Optional
    } for i, row in enumerate(ds)]

def run_full_pipeline(model_name: str, hf_dataset: str, encoder_name: str = "all-MiniLM-L6-v2", 
                     output_dir: str = "results", num_samples: int = 1000):
    """Complete pipeline: Load → Run → Save"""
    print(f"Starting pipeline: {model_name} on {hf_dataset}")
    
    # 1. Load dataset
    dataset = load_hf_dataset(hf_dataset, num_samples=num_samples)
    
    # 2. Create config
    config = benchmarking_config(
        model_name=model_name,
        encoder_name=encoder_name,
        dataset_name=hf_dataset,
        num_samples=8,  # Fixed for your table
        temperature=0.7,
        top_p=0.9
    )
    
    # 3. Run benchmark (saves graphs + metrics)
    question_metrics = run_benchmark(dataset, config, output_dir)
    
    # 4. Generate model metrics + plots
    main(model_name, hf_dataset, output_dir)
    
    print(f"COMPLETE! Check results/{hf_dataset}_{model_name.split('/')[-1]}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reasoning Graph Benchmark")
    parser.add_argument("--model", required=True, help="HF model name (e.g., microsoft/DialoGPT-medium)")
    parser.add_argument("--dataset", required=True, help="HF dataset (e.g., gsm8k)")
    parser.add_argument("--encoder", default="all-MiniLM-L6-v2", help="Sentence encoder")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="Dataset size")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        model_name=args.model,
        hf_dataset=args.dataset,
        encoder_name=args.encoder,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
