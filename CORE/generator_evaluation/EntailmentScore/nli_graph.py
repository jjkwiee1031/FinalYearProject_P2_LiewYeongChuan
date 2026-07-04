import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')
RAW_RESULTS_CSV = os.path.join(RESULT_DIR, "nli_raw_results.csv")

def generate_nli_graphs():
    print(f"Loading raw results from: {RAW_RESULTS_CSV}")
    if not os.path.exists(RAW_RESULTS_CSV):
        print(f"Error: Raw NLI results file not found at {RAW_RESULTS_CSV}")
        print("Please run 'nli.py' first to generate raw evaluation data.")
        return

    eval_df = pd.read_csv(RAW_RESULTS_CSV)
    
    if 'LLM_Model' in eval_df.columns:
        eval_df = eval_df[~eval_df['LLM_Model'].str.endswith('_claims', na=False)]
    
    score_cols = [col for col in eval_df.columns if col.endswith('_score') and col != 'combined_score']
    eval_df['combined_score'] = eval_df[score_cols].mean(axis=1)

    print("Aggregating NLI Metrics...")
    
    agg_dict = {}
    for col in score_cols:
        model_name = col[:-6].upper()
        agg_dict[f"{model_name}_Entailment"] = eval_df.groupby('LLM_Model')[col].mean()
    agg_dict['Combined_Entailment'] = eval_df.groupby('LLM_Model')['combined_score'].mean()
    agg_df = pd.DataFrame(agg_dict).reset_index()

    year_agg_dict = {}
    for col in score_cols:
        model_name = col[:-6].upper()
        year_agg_dict[f"{model_name}_Entailment"] = eval_df.groupby(['LLM_Model', 'Year_Group'])[col].mean()
    year_agg_dict['Combined_Entailment'] = eval_df.groupby(['LLM_Model', 'Year_Group'])['combined_score'].mean()
    year_agg_df = pd.DataFrame(year_agg_dict).reset_index()

    label_counts = eval_df.groupby('LLM_Model')['majority_label'].value_counts().unstack(fill_value=0)
    label_percentages = eval_df.groupby('LLM_Model')['majority_label'].value_counts(normalize=True).unstack(fill_value=0) * 100
    
    for lbl in ['entailment', 'neutral', 'contradiction', 'Ambiguous']:
        if lbl not in label_percentages.columns:
            label_percentages[lbl] = 0.0
            label_counts[lbl] = 0
            
    label_cols_ordered = ['entailment', 'neutral', 'contradiction', 'Ambiguous']
    label_percentages = label_percentages[label_cols_ordered].reset_index()
    label_counts = label_counts[label_cols_ordered].reset_index()

    year_label_percentages = eval_df.groupby(['LLM_Model', 'Year_Group'])['majority_label'].value_counts(normalize=True).unstack(fill_value=0) * 100
    for lbl in ['entailment', 'neutral', 'contradiction', 'Ambiguous']:
        if lbl not in year_label_percentages.columns:
            year_label_percentages[lbl] = 0.0
    year_label_percentages = year_label_percentages[label_cols_ordered].reset_index()

    # Form text report content
    report_content = "==================================================\n"
    report_content += "NATURAL LANGUAGE INFERENCE (NLI) EVALUATION REPORT\n"
    report_content += "==================================================\n\n"

    report_content += "--- 1. Average Entailment Scores (including Combined) ---\n"
    report_content += agg_df.to_string(index=False)
    report_content += "\n\n"

    report_content += "--- 2. Majority Vote Label Distribution (%) ---\n"
    report_content += label_percentages.to_string(index=False)
    report_content += "\n\n"

    report_content += "--- 3. Majority Vote Label Counts ---\n"
    report_content += label_counts.to_string(index=False)
    report_content += "\n\n"

    report_content += "--- 4. Average Entailment Scores Sub-grouped by Years ---\n"
    report_content += year_agg_df.to_string(index=False)
    report_content += "\n\n"

    report_content += "--- 5. Majority Vote Label Distribution (%) Sub-grouped by Years ---\n"
    report_content += year_label_percentages.to_string(index=False)
    report_content += "\n\n"

    print("\n" + "="*50)
    print("NLI Evaluation Summary Averages (Entailment Score)")
    print("="*50)
    print(agg_df.to_string(index=False))
    print("\n" + "="*50)
    print("Majority Vote Label Distribution (%)")
    print("="*50)
    print(label_percentages.to_string(index=False))
    print("\n" + "="*50)

    report_path = os.path.join(RESULT_DIR, "nli_metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved textual report to {report_path}")

    print("\nGenerating NLI charts...")
    
    nli_cols = [col for col in agg_df.columns if col != 'LLM_Model']
    num_bars = len(nli_cols)
    
    # Chart 1: Grouped Bar Chart of Average Entailment Scores
    plt.figure(figsize=(12, 6))
    x = np.arange(len(agg_df['LLM_Model']))
    width = 0.8 / num_bars
    
    # Color map for the bars
    import matplotlib.cm as cm
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, num_bars))
    
    for idx, col in enumerate(nli_cols):
        vals = agg_df[col].tolist()
        label_name = col.split('_')[0]
        # Position offset: center the bars around x
        offset = (idx - (num_bars - 1) / 2.0) * width
        plt.bar(x + offset, vals, width, label=label_name, color=colors[idx], edgecolor='black')
        for i, v in enumerate(vals):
            plt.text(i + offset, v + 0.01, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')
                
    plt.title('Average NLI Entailment Score by Generator LLM')
    plt.xlabel('Generator LLM Model')
    plt.ylabel('Average Entailment Score')
    plt.xticks(x, agg_df['LLM_Model'], rotation=15)
    plt.ylim(0, 1.1)
    plt.legend(title='NLI Evaluator')
    plt.tight_layout()
    
    chart_path = os.path.join(RESULT_DIR, "nli_average_entailment_scores.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved average entailment score comparison chart to: {chart_path}")

    # Chart 2: Stacked Bar Chart for Majority Label Distribution (%)
    plt.figure(figsize=(10, 6))
    models = label_percentages['LLM_Model'].tolist()
    ent = label_percentages['entailment'].values
    neu = label_percentages['neutral'].values
    con = label_percentages['contradiction'].values
    amb = label_percentages['Ambiguous'].values
    
    plt.bar(models, ent, label='Entailment', color='#2ecc71', edgecolor='black')
    plt.bar(models, neu, bottom=ent, label='Neutral', color='#f1c40f', edgecolor='black')
    plt.bar(models, con, bottom=ent+neu, label='Contradiction', color='#e74c3c', edgecolor='black')
    plt.bar(models, amb, bottom=ent+neu+con, label='Ambiguous', color='#9b59b6', edgecolor='black')
    
    # Annotate percentages
    for i in range(len(models)):
        y_offset = 0
        for val in [ent[i], neu[i], con[i], amb[i]]:
            if val > 2.0:
                plt.text(i, y_offset + val/2, f'{val:.1f}%', ha='center', va='center', fontsize=9, color='black', fontweight='bold')
            y_offset += val
            
    plt.title('Majority Vote NLI Label Distribution by Generator LLM')
    plt.xlabel('Generator LLM Model')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 105)
    plt.legend(title='NLI Classification', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    dist_chart_path = os.path.join(RESULT_DIR, "nli_majority_label_distributions.png")
    plt.savefig(dist_chart_path)
    plt.close()
    print(f"Saved label distribution chart to: {dist_chart_path}")

    # Charts 3+: Grouped Bar Chart by Year Group for each NLI model and Combined
    for col in nli_cols:
        plt.figure(figsize=(12, 6))
        pivot_df = year_agg_df.pivot(index='Year_Group', columns='LLM_Model', values=col)
        pivot_df.plot(kind='bar', figsize=(12, 6), edgecolor='black', rot=15)
        
        nli_name = col.split('_')[0]
        plt.title(f'NLI Entailment Score by Year Group and Generator LLM ({nli_name})')
        plt.ylabel('Average Entailment Score')
        plt.ylim(0, 1.1)
        plt.legend(title='Generator LLM', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        year_chart_path = os.path.join(RESULT_DIR, f"nli_{nli_name.lower()}_year_scores.png")
        plt.savefig(year_chart_path)
        plt.close()
        print(f"Saved year-grouped chart for {nli_name} to: {year_chart_path}")

    print("\n[Success] Graph generation and aggregation complete!")

if __name__ == "__main__":
    generate_nli_graphs()
