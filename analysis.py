import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import pickle

def load_all_results(results_dir: str = "results") -> pd.DataFrame:
    """Load ALL model+dataset results → master DataFrame"""
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
                # Assuming ModelMetrics has greedy_accuracy or we compute from question_metrics
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
                
                print(f"Loaded {model_dir.name}/{dataset_dir.name}")
                
            except Exception as e:
                print(f"Error loading {dataset_dir}: {e}")
    
    if not all_results:
        print("\nNo valid results found! Check your results directory structure.")
        print(f"Expected: {results_dir}/{{model}}/{{dataset}}/model_metrics.pkl")
    
    return pd.DataFrame(all_results)

def create_visualizations(df: pd.DataFrame, output_dir: str = "analysis"):
    """Generate ALL plots + table"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    #2D: Entropy vs Exploration Gain
    fig2d = px.scatter(df, x='avg_entropy', y='delta_acc', 
                      size='num_questions', color='model',
                      hover_data=['dataset', 'greedy_acc', 'multi_acc'],
                      title="Path Entropy vs Exploration Gain")
    fig2d.write_html(output_path / "entropy_vs_exploration.html")
    
    # 3D: Entropy × Gain × Dataset
    fig3d = px.scatter_3d(df, x='avg_entropy', y='delta_acc', z='dataset',
                         color='model', size='num_questions',
                         title="3D: Entropy × Gain × Dataset")
    fig3d.write_html(output_path / "3d_analysis.html")
    
    # Summary Table
    table_df = df.groupby('model').agg({
        'greedy_acc': 'mean',
        'multi_acc': 'mean', 
        'delta_acc': 'mean',
        'avg_entropy': 'mean'
    }).round(3)
    
    table_df['∆Acc'] = table_df['delta_acc'].apply(lambda x: f"+{x:.2f}")
    table_df['Hm'] = table_df['avg_entropy'].round(2)
    
    # Format for paper
    table_df = table_df[['greedy_acc', 'multi_acc', '∆Acc', 'Hm']]
    table_df.columns = ['Acc_greedy', 'Acc_multi(16)', '∆Acc(16)', 'Hm']
    
    table_df.to_csv(output_path / "summary_table.csv")
    print("\n SUMMARY TABLE:")
    print(table_df.to_markdown())
    
    return fig2d, fig3d, table_df

if __name__ == "__main__":
    df = load_all_results("results")
    fig2d, fig3d, table = create_visualizations(df)  # Changed: unpack 3 values
    
    print(f"\n✓ Analysis complete! Check 'analysis/' folder")
    print(f"{len(df)} model-dataset combinations analyzed")

