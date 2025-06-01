from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the data
file_path = '/Users/Nufail/Desktop/THESIS/Ablation_results/ragas_eval_log_results copy.xlsx'
df = pd.read_excel(file_path, sheet_name='ragas_eval_log')

# Set style
sns.set(style="whitegrid")
palette = sns.color_palette("husl", len(df['llm_model'].unique()))

# Create a combined configuration column
df['Configuration'] = df['llm_model'] + ' | ' + df['embedding_model'] + ' | ' + df['rerank_model']

# Plot 1: Faithfulness by Configuration
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Configuration', y='faithfulness', 
            hue='llm_model', 
            palette=palette)
plt.title('Faithfulness by Configuration')
plt.ylabel('Faithfulness')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Move the legend outside the plot
plt.legend(title='LLM Model', bbox_to_anchor=(1.05, 1), loc='upper left')
# Save the plot
plt.savefig('Faithfulness_by_Configuration_test.png', bbox_inches='tight')
plt.close()

# Plot 2: FacCor and AnsCor by Configuration
df_melted = df.melt(id_vars=['Configuration', 'llm_model'], 
                    value_vars=['avg_factual_correctness', 'avg_answer_correctness'],
                    var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Configuration', y='Score', hue='Metric')
plt.title('Factual and Answer Correctness by Configuration')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Move the legend outside the plot
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
# Save the plot with adjusted legend
plt.savefig('FacCor_AnsCor_by_Configuration_adjusted.png', bbox_inches='tight')
plt.close()

#########################################################
# Step 1: Map LLM sizes
model_size_mapping = {
    'Llama-3.1-8B': 8,
    'Llama-3.2-3B': 3,
    'Falcon3-7B': 7,
    'Qwen3-8B': 8,
    'Qwen2.5-7B': 7,
    'Deepseek-R1-8B': 8,
    'Llama-3.1-8B-Q8': 4,
    
}
df['llm_size'] = df['llm_model'].map(model_size_mapping)

# Marker mapping
marker_map = {
    8: 'o',   # Circle
    7: 's',   # Square
    4: 'D',   # Diamond
    3: '^',   # Triangle
}

plt.figure(figsize=(10, 6))

# Normalize latency
norm = plt.Normalize(df['avg_latency_per_answer'].min(), df['avg_latency_per_answer'].max())
cmap = plt.cm.coolwarm

# Plot each llm_size group with its own marker
for size, marker in marker_map.items():
    subset = df[df['llm_size'] == size]
    plt.scatter(subset['estimated_monthly_cost'], subset['faithfulness'],
                c=subset['avg_latency_per_answer'], cmap=cmap, norm=norm,
                s=size*50, marker=marker, alpha=0.7, edgecolor='k', label=f'{size}B')

plt.title('Faithfulness vs Cost with LLM Size as Shape and Latency as Color')
plt.xlabel('Estimated Monthly Cost (€)')
plt.ylabel('Faithfulness')

# Add colorbar
scatter = plt.scatter([], [], c=[], cmap=cmap, norm=norm)  # Dummy scatter for colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Avg Latency per Answer (Sec)')

# Add legend for shapes (LLM sizes)
handles = [Line2D([0], [0], marker=marker_map[size], color='w', label=f'{size}B',
                  markerfacecolor='gray', markersize=10, markeredgecolor='k') for size in marker_map]
plt.legend(handles=handles, title='LLM Size', bbox_to_anchor=(1.20, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.savefig('Faithfulness_Cost_Colorbar.png', bbox_inches='tight')
plt.close()

#############################################

# Melt data for context_precision and context_recall
df_melted_context = df.melt(id_vars=['Configuration', 'llm_model'], 
                            value_vars=['context_precision', 'context_recall'],
                            var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted_context, x='Configuration', y='Score', hue='Metric')
plt.title('Context Precision and Recall by Configuration')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Move legend outside plot
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save the plot
plt.savefig('Context_Precision_Recall_by_Configuration.png', bbox_inches='tight')
plt.close()
#############################################


