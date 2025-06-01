import pandas as pd
import krippendorff
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Excel data
file_path = '/Users/Nufail/Desktop/THESIS/UAT_RESULTS/User Acceptance Testing_ RAG Chatbot Evaluation.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
df.columns = df.columns.str.strip()

# Define rubric columns
rubric_columns = {
    'Correctness': [
        'How factually correct was the chatbot’s response for task 1?',
        'How factually correct was the chatbot’s response for task 2?',
        'How factually correct was the chatbot’s response for task 3?',
        'How factually correct was the chatbot’s response for task 4?',
        'How factually correct was the chatbot’s response for task 5?'
    ],
    'Completeness': [
        'The answer fully addresses all aspects of the prompt for task 1',
        'The answer fully addresses all aspects of the prompt for task 2',
        'The answer fully addresses all aspects of the prompt for task 3',
        'The answer fully addresses all aspects of the prompt for task 4',
        'The answer fully addresses all aspects of the prompt for task 5'
    ],
    'Clarity': [
        'The answer is clearly worded, well-structured, and easy to understand for task 1',
        'The answer is clearly worded, well-structured, and easy to understand for task 2',
        'The answer is clearly worded, well-structured, and easy to understand for task 3',
        'The answer is clearly worded, well-structured, and easy to understand for task 4',
        'The answer is clearly worded, well-structured, and easy to understand for task 5'
    ]
}

# Prepare output containers
results = []

# Function to compute Krippendorff's alpha per team and dimension
def compute_kripp_alpha(team, data):
    df_team = data[data['Which Team do you belong to'] == team]
    for dimension, cols in rubric_columns.items():
        # Clean and map values
        mapped_df = df_team[cols].apply(lambda col: col.map(lambda x: str(x).strip() if pd.notnull(x) else x))
        mapped_df = mapped_df.replace({'Very Poor':1, 'Poor':2, 'fair':3, 'Good':4, 'Excellent':5}).astype(float)
        
        # Convert to matrix with NaNs for missing
        matrix = mapped_df.T.values
        
        if matrix.size > 0 and not np.isnan(matrix).all():
            alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='ordinal')
            print(f"{team} - {dimension}: Krippendorff's alpha = {alpha:.3f}")
            results.append({'Team': team, 'Dimension': dimension, 'Krippendorff Alpha': alpha})
        else:
            print(f"{team} - {dimension}: No valid data.")
            results.append({'Team': team, 'Dimension': dimension, 'Krippendorff Alpha': np.nan})

# Compute for both teams
compute_kripp_alpha('Team Data', df)
compute_kripp_alpha('Team BI', df)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_dir = '/Users/Nufail/Desktop/THESIS/UAT_RESULTS/krippendorff'
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(f'{output_dir}/Krippendorff_Alpha_by_Team_and_Dimension.csv', index=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Dimension', y='Krippendorff Alpha', hue='Team')
plt.title("Krippendorff's Alpha by Team and Dimension")
plt.ylim(-1, 1)
plt.ylabel("Krippendorff's Alpha")
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig(f'{output_dir}/Krippendorff_Alpha_Barplot.png')
plt.close()

print(f"\nAnalysis complete. Results saved to {output_dir}")