import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, 'result'))
RAW_RESULTS_CSV = os.path.join(RESULT_DIR, "refusal_raw_results.csv")

def generate_refusal_graphs():
    print("\n" + "="*50)
    print("RUNNING REFUSAL RATE GRAPH GENERATOR")
    print("="*50)
    
    print(f"Loading raw results from: {RAW_RESULTS_CSV}")
    if not os.path.exists(RAW_RESULTS_CSV):
        print(f"Error: Raw refusal results file not found at {RAW_RESULTS_CSV}")
        print("Please run 'refusal.py' first to generate raw evaluation data.")
        return

    eval_df = pd.read_csv(RAW_RESULTS_CSV)

    overall_keyword = eval_df.groupby('LLM_Model')['Is_Refusal_Keyword'].mean() * 100
    overall_nli = eval_df.groupby('LLM_Model')['Is_Refusal_NLI'].mean() * 100
    total_counts = eval_df.groupby('LLM_Model').size()

    overall_df = pd.DataFrame({
        'Total_Questions': total_counts,
        'Refusal_Rate_Keyword(%)': overall_keyword,
        'Refusal_Rate_NLI(%)': overall_nli
    }).reset_index()

    embed_keyword = eval_df.groupby(['LLM_Model', 'Embedding_Model'])['Is_Refusal_Keyword'].mean() * 100
    embed_nli = eval_df.groupby(['LLM_Model', 'Embedding_Model'])['Is_Refusal_NLI'].mean() * 100
    embed_counts = eval_df.groupby(['LLM_Model', 'Embedding_Model']).size()

    embed_df = pd.DataFrame({
        'Total_Questions': embed_counts,
        'Refusal_Rate_Keyword(%)': embed_keyword,
        'Refusal_Rate_NLI(%)': embed_nli
    }).reset_index()

    qset_keyword = eval_df.groupby(['LLM_Model', 'Question_Set'])['Is_Refusal_Keyword'].mean() * 100
    qset_nli = eval_df.groupby(['LLM_Model', 'Question_Set'])['Is_Refusal_NLI'].mean() * 100
    qset_counts = eval_df.groupby(['LLM_Model', 'Question_Set']).size()

    qset_df = pd.DataFrame({
        'Total_Questions': qset_counts,
        'Refusal_Rate_Keyword(%)': qset_keyword,
        'Refusal_Rate_NLI(%)': qset_nli
    }).reset_index()

    year_keyword = eval_df.groupby(['LLM_Model', 'Year_Group'])['Is_Refusal_Keyword'].mean() * 100
    year_nli = eval_df.groupby(['LLM_Model', 'Year_Group'])['Is_Refusal_NLI'].mean() * 100
    year_counts = eval_df.groupby(['LLM_Model', 'Year_Group']).size()

    year_df = pd.DataFrame({
        'Total_Questions': year_counts,
        'Refusal_Rate_Keyword(%)': year_keyword,
        'Refusal_Rate_NLI(%)': year_nli
    }).reset_index()

    summary_csv_path = os.path.join(RESULT_DIR, "refusal_rates_summary.csv")
    qset_df.to_csv(summary_csv_path, index=False)
    print(f"Saved aggregated question set refusal rates to: {summary_csv_path}")

    # Form text report content
    report_content = "========================================================\n"
    report_content += "REFUSAL RATE EVALUATION REPORT (UNANSWERABLE QUESTIONS)\n"
    report_content += "========================================================\n\n"

    report_content += "--- 1. Overall Refusal Rates by Generator LLM ---\n"
    report_content += overall_df.to_string(index=False)
    report_content += "\n\n"

    report_content += "--- 2. Refusal Rates by Generator LLM & Embedding Model ---\n"
    report_content += embed_df.to_string(index=False)
    report_content += "\n\n"

    report_content += "--- 3. Refusal Rates by Generator LLM & Year Group ---\n"
    report_content += year_df.to_string(index=False)
    report_content += "\n\n"

    report_content += "--- 4. Detailed Refusal Rates per Question Set (Top 30 Rows) ---\n"
    report_content += qset_df.head(30).to_string(index=False)
    report_content += "\n\n"

    print(report_content)

    report_path = os.path.join(RESULT_DIR, "refusal_metrics_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved textual report to: {report_path}")

    print("\nGenerating refusal charts...")

    # Chart 1: Bar Chart of Overall Refusal Rates (Keyword vs NLI)
    fig, ax = plt.subplots(figsize=(10, 6))
    models = overall_df['LLM_Model'].tolist()
    x = np.arange(len(models))
    width = 0.35  # width of the bars

    rects1 = ax.bar(x - width/2, overall_df['Refusal_Rate_Keyword(%)'], width, label='Keyword-based', color='#3498db', edgecolor='black')
    rects2 = ax.bar(x + width/2, overall_df['Refusal_Rate_NLI(%)'], width, label='NLI-based', color='#2ecc71', edgecolor='black')

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    ax.set_title('Overall Refusal Rates on Unanswerable Questions by Generator LLM')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlabel('Generator LLM Model')
    ax.set_ylabel('Refusal Rate (%)')
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()

    chart1_path = os.path.join(RESULT_DIR, "refusal_rates_by_generator.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"Saved generator comparison chart to: {chart1_path}")

    # Chart 2: Grouped Bar Chart of NLI Refusal Rate by Embedding Model
    plt.figure(figsize=(12, 6))
    pivot_embed = embed_df.pivot(index='Embedding_Model', columns='LLM_Model', values='Refusal_Rate_NLI(%)')
    pivot_embed.plot(kind='bar', edgecolor='black', rot=15, colormap='viridis', figsize=(12, 6))
    
    plt.title('NLI-Based Refusal Rates by Embedding Model and Generator LLM')
    plt.ylabel('Refusal Rate (%)')
    plt.ylim(0, 110)
    plt.legend(title='Generator LLM', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    chart2_path = os.path.join(RESULT_DIR, "refusal_rates_by_embedding.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"Saved embedding comparison chart to: {chart2_path}")

    # Chart 3: Grouped Bar Chart of NLI Refusal Rate by Year Group
    plt.figure(figsize=(12, 6))
    pivot_year = year_df.pivot(index='Year_Group', columns='LLM_Model', values='Refusal_Rate_NLI(%)')
    pivot_year.plot(kind='bar', edgecolor='black', rot=15, colormap='coolwarm', figsize=(12, 6))
    
    plt.title('NLI-Based Refusal Rates by Year Group and Generator LLM')
    plt.ylabel('Refusal Rate (%)')
    plt.ylim(0, 110)
    plt.legend(title='Generator LLM', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    chart3_path = os.path.join(RESULT_DIR, "refusal_rates_by_year.png")
    plt.savefig(chart3_path)
    plt.close()
    print(f"Saved year-grouped chart to: {chart3_path}")

    print("\n[Success] Refusal rate evaluation graphs and report completed successfully!")

if __name__ == "__main__":
    generate_refusal_graphs()
