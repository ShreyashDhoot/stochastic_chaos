#!/usr/bin/env python3
import argparse
import datasets
import re
import pandas as pd
import torch
import numpy as np 
import pickle
import os
from enum import Enum
from typing import List, Dict,Tuple,Optional, Union
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv

from graph_structure import ReasoningGraph,Node, OutcomeType,benchmarking_config,QuestionMetrics,ModelMetrics,BenchmarkResults
from models import load_lm, load_encoder
from generation import generate_greedy, generate_multi_sample
from preprocess import split_into_steps, encode_steps
from dag_build import add_chain_to_graph
from outcome import label_leaf_outcomes
from metrics import (
    greedy_support_ratio, 
    collapse_failure_rate,
    extract_question_metrics,
    exploration_gain
)

from answer_parser import extract_final_answer, answers_match, normalize_answer
from graphviz import Digraph
from IPython.display import Image, display


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
    print("Building reasoning graph...")
    graph = ReasoningGraph(question_id=question.get('id', 0),question_text=question_text,ground_truth_answer=ground_truth_answer,similarity_threshold=config.similarity_threshold,nearmiss_overlap_threshold=config.nearmiss_threshold)

    #build reasoning graph for greedy sampling 
    greedy_steps=split_into_steps(greedy_answer)
    greedy_embs = encode_steps(encoder, greedy_steps)
    add_chain_to_graph(graph, greedy_steps, greedy_embs,is_greedy=True,logprob=greedy_logprobs)

    #build reasoning graph for multiplesampling 
    for i,(text,logprob) in enumerate(zip(sample_texts,sample_logprobs)):
        steps=split_into_steps(text)
        if not steps:
            continue
        embs=encode_steps(encoder,steps)
        add_chain_to_graph(graph,steps,embs,is_greedy=False,logprob=logprob)


    #breaking in steps and encoding the ground truth answer and seperating 
    #this needs to be list of np.ndarray come back while writing the main 
    gt_step_embs = encode_steps(encoder, ground_truth_steps) if ground_truth_steps else None
    
    #labelling the leaf outcomes 
    label_leaf_outcomes(graph,ground_truth_answer,gt_step_embs)
   # dot= plot_reasoning_graph(graph)
    #png_path = dot.render("/kaggle/working/reasoning_graph_q0", format="png", cleanup=True)
    #display(Image(png_path))
    print("GRAPH CREATED\n")
    return graph

def run_benchmark(dataset: List[Dict], config:benchmarking_config, output_dir: str = "results") -> list[QuestionMetrics]:
    """Run complete benchmark"""
    model_short = config.model_name.split('/')[-1]
    dataset_safe = config.dataset_name.replace('/', '-')
    output_path = Path(output_dir) / model_short / dataset_safe
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {config.model_name}")
    model, tokenizer = load_lm(config.model_name)
    
    print(f"Loading encoder: {config.encoder_name}")
    encoder = load_encoder(config.encoder_name)

    question_metrics_list=[]

    for i, question in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}]")
        print("run for question function call")
        graph = run_for_question(model, tokenizer, encoder,config,question)
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
    model_short = model_name.split('/')[-1]
    dataset_safe = hf_dataset.replace('/', '-')

    # 1. Load QuestionMetrics (no graphs needed!)
    output_path = Path(output_dir) / model_short / dataset_safe
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
        question_metrics=question_metrics_list,
    )
    
    # 3. Save ModelMetrics
    with open(output_path / "model_metrics.pkl", "wb") as f:
        pickle.dump(model_metrics, f)
    
    print(f" Complete analysis for {model_name} on {hf_dataset}")
    return model_metrics

def load_hf_dataset(dataset_name: str, split: str = "test",dataset_config: str | None = None, num_samples: int = 1000) -> List[Dict]:

    """Load HF dataset → List[Dict]"""

    # StrategyQA only has 'train' split, adjust if needed
    if 'strategy-qa' in dataset_name.lower() and split == 'test':
        split = 'train'
        print(f"Note: StrategyQA only has 'train' split, using that instead")

    print(f"Loading {dataset_name}/{dataset_config} ({split}[:{num_samples}])")
    if dataset_config is None:
        ds = datasets.load_dataset(dataset_name, split=f"{split}[:{num_samples}]")
    else:
        ds = datasets.load_dataset(dataset_name, dataset_config, split=f"{split}[:{num_samples}]")
    
    print("dataset loaded")
    
    # Detect dataset format and standardize
    result = []
    for i, row in enumerate(ds):
        # StrategyQA format (tasksource/strategy-qa)
        if 'qid' in row and 'facts' in row and 'decomposition' in row:
            # Build labeled composite question
            parts = []
    
            description = row.get('description', '').strip()
            if description:
                parts.append(f"Description: {description}")
    
            facts_list = row.get('facts', [])
            if facts_list:
                facts_text = ' '.join(facts_list).strip()
                parts.append(f"Facts: {facts_text}")
    
            question = row.get('question', '').strip()
            if question:
                parts.append(f"Question: {question}")
    
            composite_question = ' '.join(parts)
            
            result.append({
                'id': i,
                'question': composite_question,
                'answer': str(row['answer']).lower(),  # 'true' or 'false'
                'solution_steps': []  # No intermediate reasoning
            })

        # SVAMP format (ChilleD/SVAMP)
        elif 'question_concat' in row:
            result.append({
                'id': i,
                'question': row['question_concat'],
                'answer': str(row['Answer']),
                'solution_steps': [] 
            })
        # GSM8K format (standard 'question' column)
        elif 'question' in row:
            result.append({
                'id': i,
                'question': row['question'],
                'answer': row.get('answer', row.get('final_answer', '')),
                'solution_steps': row.get('solution', [])
            })
        else:
            raise ValueError(f"Unknown dataset format. Available columns: {list(row.keys())}")
    
    return result

def run_full_pipeline(model_name: str, hf_dataset: str, dataset_config: str | None = None,encoder_name: str = "all-MiniLM-L6-v2", 
                     output_dir: str = "results", num_samples: int = 1000):
    """Complete pipeline: Load → Run → Save"""
    print(f"Starting pipeline: {model_name} on {hf_dataset}")
    
    # 1. Load dataset
    dataset = load_hf_dataset(hf_dataset,dataset_config=args.dataset_config, num_samples=num_samples)
    
    # 2. Create config
    config = benchmarking_config(
        model_name=model_name,
        encoder_name=encoder_name,
        dataset_name=hf_dataset,
        num_samples=8,  
        temperature=0.7,
        top_p=0.9
    )
    
    # 3. Run benchmark (saves graphs + metrics)
    print("Running benchmark... function call")
    question_metrics = run_benchmark(dataset, config, output_dir)
    
    main(model_name, hf_dataset, output_dir)
    
    print(f"COMPLETE! Check results/{hf_dataset}_{model_name.split('/')[-1]}/")

if __name__ == "__main__":
    #### HF-login for llama models#########
    load_dotenv() 
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged into HuggingFace")
    else:
        # Option 2: Fallback to interactive login
        try:
            login()  # Will prompt for token if not already logged in
            print("Logged into HuggingFace")
        except Exception as e:
            print(f"HuggingFace login failed: {e}")
            print("Some models may not load without authentication")

    parser = argparse.ArgumentParser(description="Reasoning Graph Benchmark")
    parser.add_argument("--model", required=True, help="HF model name (e.g., microsoft/DialoGPT-medium)")
    parser.add_argument("--dataset", required=True, help="HF dataset (e.g., gsm8k)")
    parser.add_argument("--dataset-config", default=None, help="HF dataset config name (e.g., main)")
    parser.add_argument("--encoder", default="all-MiniLM-L6-v2", help="Sentence encoder")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="Dataset size")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        model_name=args.model,
        hf_dataset=args.dataset,
        dataset_config=args.dataset_config,
        encoder_name=args.encoder,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
