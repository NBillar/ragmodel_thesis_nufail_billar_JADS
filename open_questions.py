import pandas as pd
from collections import Counter
import os
import matplotlib.pyplot as plt

# Load data
file_path = '/Users/Nufail/Desktop/THESIS/UAT_RESULTS/User Acceptance Testing_ RAG Chatbot Evaluation.xlsx'
output_dir = '/Users/Nufail/Desktop/THESIS/UAT_RESULTS/OpenEndedAnalysis'
os.makedirs(output_dir, exist_ok=True)
df = pd.read_excel(file_path, sheet_name='Sheet1')
df.columns = df.columns.str.strip()

# Define text columns
text_cols = {
    'Helpful': 'What did you find most helpful about the chatbot?',
    'Challenges': 'What difficulties or challenges did you experience while using the chatbot?',
    'Trust': 'Did you trust the answers the chatbot provided? Why or why not?',
    'Improvements': 'What improvements would you suggest to make the chatbot more useful or easier to use?',
    'Comments': 'Any additional comments or thoughts about using the chatbot?'
}

# Define refined themes and keywords
theme_keywords = {
    'Trust and Reliability': ['trust', 'reliable', 'provenance', 'confidence'],
    'Usefulness and Efficiency': ['useful', 'helpful', 'task', 'solve', 'efficient'],
    'Clarity and Explanation': ['clear', 'confusing', 'understand', 'vague', 'explanation'],
    'Ease of Use and Interaction': ['easy', 'simple', 'intuitive', 'chatgpt', 'interaction'],
    'Suggested Improvements': ['improve', 'fix', 'faster', 'update', 'expand', 'add'],
    'Unique Observations': ['junior', 'dyslexic', 'github', 'agent', 'documentation']
}

# Initialize summary
theme_counts_total = Counter()
theme_counts_by_question = {question: Counter() for question in text_cols}

# Analyze each question separately
for question_key, col_name in text_cols.items():
    responses = df[col_name].dropna().str.lower().tolist()
    for response in responses:
        for theme, keywords in theme_keywords.items():
            if any(keyword in response for keyword in keywords):
                theme_counts_total[theme] += 1
                theme_counts_by_question[question_key][theme] += 1

# Save total counts
pd.DataFrame.from_dict(theme_counts_total, orient='index', columns=['Total_Count']) \
    .to_csv(os.path.join(output_dir, 'Theme_Frequencies_Total.csv'))

# Save counts by question
theme_counts_question_df = pd.DataFrame(theme_counts_by_question).fillna(0).astype(int)
theme_counts_question_df.to_csv(os.path.join(output_dir, 'Theme_Frequencies_ByQuestion.csv'))

# Optional: Generate bar plot of total theme frequencies
plt.figure(figsize=(10,6))
theme_counts_question_df.sum(axis=1).sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Total Theme Frequencies Across Open-Ended Questions')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'OpenEnded_Theme_Frequency_Barplot.png'))
plt.close()

print(f"✅ Open-ended theme analysis completed. Results saved to {output_dir}")