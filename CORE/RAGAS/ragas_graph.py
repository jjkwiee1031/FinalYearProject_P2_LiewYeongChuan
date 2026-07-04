import os
import re
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "ragas_results.csv")

metrics = {

}

if os.path.exists(csv_path):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        parsed_metrics = {}
        for _, row in df.iterrows():
            name = str(row["Metric"]).strip()
            parsed_metrics[name] = {
                "before": float(row["Before"]),
                "after": float(row["After"])
            }
        
        if all(k in parsed_metrics for k in metrics.keys()):
            metrics = parsed_metrics
            print("Successfully loaded latest values from ragas_results.csv.")
    except Exception as e:
        print(f"Error reading ragas_results.csv: {e}. Using hardcoded fallback values instead.")
else:
    print("ragas_results.csv not found. Using default comparison values.")

core_labels = ["Answer Relevancy", "Faithfulness", "Context Precision", "Context Recall"]
core_keys = ["Answer Relevancy Score", "Faithfulness Score", "Context Precision Score", "Context Recall Score"]
core_before = [metrics[k]["before"] for k in core_keys]
core_after = [metrics[k]["after"] for k in core_keys]

agg_labels = ["Generator Average", "Retriever Average", "Overall RAGAS Index"]
agg_keys = ["Generator Metrics Average", "Retriever Metrics Average", "Overall RAGAS Index"]
agg_before = [metrics[k]["before"] for k in agg_keys]
agg_after = [metrics[k]["after"] for k in agg_keys]

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

x_core = np.arange(len(core_labels))
x_agg = np.arange(len(agg_labels))
width = 0.35

color_before = "#3B82F6"
color_after = "#10B981"   # Premium Emerald Green

rects1_b = ax1.bar(x_core - width/2, core_before, width, label='Before (Baseline)', color=color_before, edgecolor='none', alpha=0.9)
rects1_a = ax1.bar(x_core + width/2, core_after, width, label='After (Optimized)', color=color_after, edgecolor='none', alpha=0.9)

ax1.set_title("Core RAGAS Metrics Comparison", fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x_core)
ax1.set_xticklabels(core_labels, fontsize=11)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_ylim(0, 1.1)
ax1.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

rects2_b = ax2.bar(x_agg - width/2, agg_before, width, label='Before (Baseline)', color=color_before, edgecolor='none', alpha=0.9)
rects2_a = ax2.bar(x_agg + width/2, agg_after, width, label='After (Optimized)', color=color_after, edgecolor='none', alpha=0.9)

ax2.set_title("Aggregated Performance & Index", fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x_agg)
ax2.set_xticklabels(agg_labels, fontsize=11)
ax2.set_ylabel("Score", fontsize=12)
ax2.set_ylim(0, 1.1)
ax2.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

def autolabel(rects, ax, before_vals=None):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        label = f"{height:.4f}"
        
        # If these are 'after' bars and we have 'before' values, calculate improvement percentage
        if before_vals is not None:
            b_val = before_vals[i]
            diff = height - b_val
            pct = (diff / b_val) * 100 if b_val > 0 else 0
            if diff != 0:
                label += f"\n({pct:+.1f}%)"
        
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

# Annotate bars
autolabel(rects1_b, ax1)
autolabel(rects1_a, ax1, core_before)
autolabel(rects2_b, ax2)
autolabel(rects2_a, ax2, agg_before)

plt.suptitle("RAGAS Retrieval-Augmented Generation Evaluation Report", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

output_image_path = os.path.join(current_dir, "ragas_comparison_chart.png")
plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
print(f"\n[Success] Comparison chart successfully generated and saved to: {output_image_path}")

try:
    import pandas as pd
    
    csv_data = []
    for name, vals in metrics.items():
        before_val = vals["before"]
        after_val = vals["after"]
        diff = after_val - before_val
        pct = (diff / before_val) * 100 if before_val > 0 else 0
        
        csv_data.append({
            "Metric": name,
            "Before (Baseline)": before_val,
            "After (Optimized)": after_val,
            "Absolute Improvement": diff,
            "Percentage Improvement (%)": pct
        })
        
    df_csv = pd.DataFrame(csv_data)
    csv_output_path = os.path.join(current_dir, "ragas_comparison_metrics.csv")
    df_csv.to_csv(csv_output_path, index=False)
    print(f"[Success] Comparison metrics CSV saved to: {csv_output_path}")
except Exception as e:
    print(f"Error saving comparison metrics CSV: {e}")
