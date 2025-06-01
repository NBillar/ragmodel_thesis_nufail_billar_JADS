import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import pingouin as pg
import numpy as np
import os

# Load data
file_path = '/Users/Nufail/Desktop/THESIS/UAT_RESULTS/User Acceptance Testing_ RAG Chatbot Evaluation.xlsx'
output_dir = '/Users/Nufail/Desktop/THESIS/UAT_RESULTS/cbach'
os.makedirs(output_dir, exist_ok=True)
df = pd.read_excel(file_path, sheet_name='Sheet1')
df.columns = df.columns.str.strip()

################### TAM Analysis ##########################################
tam_mapping = {
    'PEOU': [
        'The chatbot was easy to interact with.',
        'The chatbot understood my questions accurately.'
    ],
    'PU': [
        'The chatbot helped me complete my task successfully.',
        'The information provided by the chatbot was useful for my work.'
    ],
    'Trust': [
        'I trusted the information the chatbot provided.',
        'I felt confident relying on the chatbot’s answers.'
    ],
    'ATT': [
        'I am satisfied with the overall experience using the chatbot.',
        'I would consider using this chatbot for similar tasks in the future.'
    ]
}

# Likurt scale maping
likert_mapping = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly agree': 5
}

# Calculate TAM scores and Cronbachs alpha
df_tam = pd.DataFrame()
cronbach = {}

for construct, cols in tam_mapping.items():
    # map Likert labels to numbers
    clean_data = df[cols].applymap(lambda x: str(x).strip() if pd.notnull(x) else x)
    numeric_data = clean_data.replace(likert_mapping)
    data = numeric_data.apply(pd.to_numeric, errors='coerce')
    
    # Save average scores per construct
    df_tam[construct] = data.mean(axis=1)
    df_tam['Group'] = df['Which Team do you belong to']
    
    # Cronbachs alpha calculation
    valid_data = data.dropna()
    participant_count = valid_data.shape[0]
    if participant_count > 1:
        alpha = pg.cronbach_alpha(valid_data)[0]
    else:
        alpha = np.nan
        print(f"Only {participant_count} valid participant(s) for '{construct}' – Cronbachs alpha not computed.")
    cronbach[construct] = alpha

# Save summary stats
tam_summary = df_tam.groupby('Group').agg(['mean', 'std']).round(2)
tam_summary.to_csv(os.path.join(output_dir, 'TAM_Summary_ByGroup.csv'))
pd.DataFrame(cronbach.items(), columns=['Construct', 'Cronbach_Alpha']) \
    .to_csv(os.path.join(output_dir, 'TAM_Cronbach_Alpha.csv'))

# Boxplot of TAM scores
plt.figure(figsize=(10,6))
tam_melted = df_tam.melt(id_vars='Group', var_name='Construct', value_name='Score').dropna()
# Inspect the data being plotted
print("\n Data used for boxplot (by Construct and Group)")
for construct in tam_mapping.keys():
    for group in df_tam['Group'].unique():
        subset = tam_melted[(tam_melted['Construct'] == construct) & (tam_melted['Group'] == group)]
        print(f"\nConstruct: {construct}, Group: {group}")
        print(subset['Score'].describe())  # Summary of scores
        print(subset['Score'].values)      # Raw values
sns.boxplot(data=tam_melted, x='Construct', y='Score', hue='Group', palette='Set2')
plt.title('TAM Scores by Construct and Group')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'TAM_Boxplot_ByGroup.png'))
plt.close()

################## Open-Ended Feedback ##########################################
text_cols = [
    'What did you find most helpful about the chatbot?',
    'What difficulties or challenges did you experience while using the chatbot?',
    'Did you trust the answers the chatbot provided? Why or why not?',
    'What improvements would you suggest to make the chatbot more useful or easier to use?',
    'Any additional comments or thoughts about using the chatbot?'
]
all_text = ' '.join(df[text_cols].fillna('').values.flatten().tolist()).lower()

# Word frequency
words = [word for word in all_text.split() if len(word) > 3]
word_counts = Counter(words).most_common(20)
pd.DataFrame(word_counts, columns=['Word', 'Frequency']).to_csv(os.path.join(output_dir, 'OpenEnded_TopWords.csv'))

# Word cloud
if all_text.strip():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Open-Ended Feedback Word Cloud')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'OpenEnded_WordCloud.png'))
    plt.close()


theme_keywords = {
    'Trust Concerns': ['trust', 'doubt', 'reliable', 'hallucination'],
    'Perceived Helpfulness': ['helpful', 'useful', 'solved', 'task', 'benefit'],
    'Ease of Use / Usability': ['easy', 'simple', 'quick', 'responsive'],
    'Suggestions for Improvement': ['improve', 'fix', 'add', 'change', 'update'],
    'Clarity / Confusion': ['confusing', 'clear', 'understand', 'vague']
}
theme_counts = {theme: sum(all_text.count(word) for word in words) for theme, words in theme_keywords.items()}
pd.DataFrame(theme_counts.items(), columns=['Theme', 'Count']).to_csv(os.path.join(output_dir, 'OpenEnded_ThemeCounts.csv'))

# ############ Behavioral Metrics ########################
df['TaskTime'] = df['How long did it take you to complete the chatbot tasks?']
df['Rephrased'] = df['Did you have to rephrase any prompt to get a useful answer?']
df['TaskTime'].value_counts().to_csv(os.path.join(output_dir, 'TaskTime_Summary.csv'))
df['Rephrased'].value_counts().to_csv(os.path.join(output_dir, 'Rephrased_Summary.csv'))

# Plot task times
plt.figure(figsize=(8,5))
df['TaskTime'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Task Completion Time')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'TaskTime_Barplot.png'))
plt.close()

# Plot rephrased
plt.figure(figsize=(6,4))
df['Rephrased'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Prompt Rephrasing Count')
plt.xlabel('Rephrased')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Rephrased_Barplot.png'))
plt.close()

print(f"\n UAT analysis completed (excluding Krippendorffs alpha). Outputs saved to {output_dir}")