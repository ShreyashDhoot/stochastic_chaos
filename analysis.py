import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

def load_all_results(results_dir: str = "results") -> pd.DataFrame:
    """Load ALL model+dataset results → master DataFrame"""
    results_dir = Path(results_dir)
    all_results = []
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            try:
                # Load model metrics
                with open(dataset_dir / "model_metrics.pkl", "rb") as f:
                    model_metrics = pickle.load(f)
                
                # Load question metrics for avg entropy
                q_df = pd.read_csv(dataset_dir / "question_metrics.csv")
                
                all_results.append({
                    'model': model_metrics.model_name.split('/')[-1],
                    'dataset': model_metrics.dataset_name,
                    'greedy_acc': model_metrics.greedy_accuracy,
                    'multi_acc': (model_metrics.num_questions - model_metrics.multi_failures) / model_metrics.num_questions,
                    'delta_acc': model_metrics.exploration_gain / model_metrics.num_questions,
                    'avg_entropy': q_df['path_entropy'].mean(),
                    'collapse_rate': model_metrics.collapse_failure,
                    'rescue_rate': model_metrics.rescue_rate,
                    'num_questions': model_metrics.num_questions
                })
                
            except FileNotFoundError:
                print(f"Skipping {dataset_dir} (missing files)")
    
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
    figs, table = create_visualizations(df)
    
    print(f"\n Analysis complete! Check 'analysis/' folder")
    print(f"{len(df)} model-dataset combinations analyzed")
