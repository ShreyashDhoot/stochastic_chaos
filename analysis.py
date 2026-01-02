import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import pickle


def load_all_results(results_dir: str = "results") -> pd.DataFrame:
    """Load ALL model+dataset results → master DataFrame (AGGREGATED)"""
    results_dir = Path(results_dir)
    all_results = []
    
    # Iterate through model directories
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Iterate through dataset directories within each model
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            metrics_path = dataset_dir / "model_metrics.pkl"
            csv_path = dataset_dir / "question_metrics.csv"
            
            if not metrics_path.exists() or not csv_path.exists():
                print(f"Skipping {dataset_dir} (missing model_metrics.pkl or question_metrics.csv)")
                continue
            
            try:
                # Load model metrics
                with open(metrics_path, "rb") as f:
                    model_metrics = pickle.load(f)
                
                # Load question metrics
                q_df = pd.read_csv(csv_path)
                
                # Calculate multi-sample accuracy
                greedy_correct = sum(q_df['greedy_outcome'] == 'OutcomeType.CORRECT')
                multi_correct = sum(q_df['num_correct_leaves'] > 0)
                
                all_results.append({
                    'model': model_dir.name,
                    'dataset': dataset_dir.name,
                    'greedy_acc': greedy_correct / len(q_df) if len(q_df) > 0 else 0,
                    'multi_acc': multi_correct / len(q_df) if len(q_df) > 0 else 0,
                    'delta_acc': model_metrics.exploration_gain / model_metrics.num_questions if model_metrics.num_questions > 0 else 0,
                    'avg_entropy': float(q_df['path_entropy'].mean()) if not q_df.empty else 0,
                    'collapse_rate': model_metrics.collapse_failure if hasattr(model_metrics, 'collapse_failure') else 0,
                    'num_questions': model_metrics.num_questions
                })
                
                print(f"✓ Loaded {model_dir.name}/{dataset_dir.name}")
                
            except Exception as e:
                print(f"Error loading {dataset_dir}: {e}")
    
    if not all_results:
        print("\n⚠️ No valid results found! Check your results directory structure.")
        print(f"Expected: {results_dir}/{{model}}/{{dataset}}/model_metrics.pkl")
    
    return pd.DataFrame(all_results)


def create_visualizations(df: pd.DataFrame, output_dir: str = "analysis"):
    """Generate aggregate plots + table"""
    
    if df.empty:
        print("⚠️ DataFrame is empty, skipping aggregate visualizations")
        return None, None, None
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 2D: Entropy vs Exploration Gain (aggregate)
    fig2d = px.scatter(df, x='avg_entropy', y='delta_acc', 
                      size='num_questions', color='model',
                      hover_data=['dataset', 'greedy_acc', 'multi_acc'],
                      title="Aggregate: Path Entropy vs Exploration Gain")
    fig2d.write_html(output_path / "entropy_vs_exploration_aggregate.html")
    
    # 3D: Entropy × Gain × Dataset (aggregate)
    fig3d = px.scatter_3d(df, x='avg_entropy', y='delta_acc', z='dataset',
                         color='model', size='num_questions',
                         title="Aggregate: 3D Entropy × Gain × Dataset")
    fig3d.write_html(output_path / "3d_analysis_aggregate.html")
    
    # Summary Table
    table_df = df.groupby('model').agg({
        'greedy_acc': 'mean',
        'multi_acc': 'mean', 
        'delta_acc': 'mean',
        'avg_entropy': 'mean'
    }).round(3)
    
    table_df['∆Acc'] = table_df['delta_acc'].apply(lambda x: f"+{x:.2f}" if x >= 0 else f"{x:.2f}")
    table_df['Hm'] = table_df['avg_entropy'].round(2)
    
    # Format for paper
    table_df = table_df[['greedy_acc', 'multi_acc', '∆Acc', 'Hm']]
    table_df.columns = ['Acc_greedy', 'Acc_multi(16)', '∆Acc(16)', 'Hm']
    
    table_df.to_csv(output_path / "summary_table.csv")
    print("\n📊 SUMMARY TABLE:")
    print(table_df.to_markdown())
    
    return fig2d, fig3d, table_df


def create_per_question_visualization(results_dir: str = "results", output_dir: str = "analysis"):
    """Generate per-question 3D scatter like the paper (MAIN ANALYSIS)"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_question_data = []
    
    # Load question-level metrics from CSV
    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir():
            continue
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            csv_path = dataset_dir / "question_metrics.csv"
            if not csv_path.exists():
                continue
            
            q_df = pd.read_csv(csv_path)
            
            # Add model and dataset columns
            q_df['model'] = model_dir.name
            q_df['dataset'] = dataset_dir.name
            
            # Compute per-question exploration gain (binary: did multi-sampling help?)
            q_df['greedy_correct'] = (q_df['greedy_outcome'] == 'OutcomeType.CORRECT').astype(int)
            q_df['multi_correct'] = (q_df['num_correct_leaves'] > 0).astype(int)
            q_df['delta_acc'] = q_df['multi_correct'] - q_df['greedy_correct']
            
            all_question_data.append(q_df)
            print(f"✓ Loaded {len(q_df)} questions from {model_dir.name}/{dataset_dir.name}")
    
    if not all_question_data:
        print("⚠️ No per-question data found!")
        return None
    
    # Combine all questions (NO AGGREGATION!)
    df = pd.concat(all_question_data, ignore_index=True)
    
    print(f"\n📊 Total questions loaded: {len(df)}")
    print(f"   Datasets: {df['dataset'].unique()}")
    print(f"   Models: {df['model'].unique()}")
    print(f"   Δ Acc distribution: {df['delta_acc'].value_counts().to_dict()}")
    
    # 3D scatter: each point is one question
    fig3d = px.scatter_3d(
        df, 
        x='path_entropy',          # Per-question entropy
        y='delta_acc',             # Per-question gain (−1, 0, or +1)
        z='dataset',               # Dataset layer
        color='model',
        hover_data=['question_id', 'num_correct_leaves', 'greedy_outcome'],
        title="Per-Question Analysis: Entropy × Exploration Gain × Dataset"
    )
    
    fig3d.write_html(output_path / "per_question_3d.html")
    print(f"✓ Created per-question 3D plot: analysis/per_question_3d.html")
    
    # Optional: 2D version colored by delta_acc
    fig2d = px.scatter(
        df,
        x='path_entropy',
        y='delta_acc',
        color='dataset',
        facet_col='model',
        hover_data=['question_id', 'greedy_outcome'],
        title="Per-Question: Entropy vs Exploration Gain (faceted by model)"
    )
    fig2d.write_html(output_path / "per_question_2d_faceted.html")
    
    return fig3d, df


if __name__ == "__main__":
    print("="*60)
    print("ANALYSIS PIPELINE")
    print("="*60)
    
    # 1. Aggregate analysis (model-level summaries)
    print("\n[1/2] Loading aggregate metrics...")
    df_aggregate = load_all_results("results")
    
    if not df_aggregate.empty:
        fig2d, fig3d, table = create_visualizations(df_aggregate, output_dir="analysis")
        print(f"\n✓ Aggregate analysis complete!")
        print(f"   - {len(df_aggregate)} model-dataset combinations")
        print(f"   - Files: entropy_vs_exploration_aggregate.html, 3d_analysis_aggregate.html")
    
    # 2. Per-question analysis (like the paper)
    print("\n[2/2] Loading per-question metrics...")
    fig_per_q, df_per_q = create_per_question_visualization("results", "analysis")
    
    if df_per_q is not None:
        print(f"\n✓ Per-question analysis complete!")
        print(f"   - {len(df_per_q)} individual questions plotted")
        print(f"   - Files: per_question_3d.html, per_question_2d_faceted.html")
    
    print("\n" + "="*60)
    print("✅ ALL ANALYSIS COMPLETE! Check 'analysis/' folder")
    print("="*60)
