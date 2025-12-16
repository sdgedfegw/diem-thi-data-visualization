import pandas as pd
import numpy as np
from scipy.stats import norm

# Load the Excel file
df = pd.read_excel('mon_score_distribution_raw.xlsx', header=None)

# Find start rows for each year
start_rows = []
years = []
for row in range(len(df)):
    val = df.iloc[row, 0]
    if pd.notna(val) and str(val)[:4].isdigit() and int(str(val)[:4]) >= 2000:
        years.append(int(val))
        start_rows.append(row)

# Process each year
output = []
epsilon = 1e-9
for idx, start in enumerate(start_rows):
    year = years[idx]
    header_row = start + 1
    data_start = start + 2
    # Find the total row
    total_row = None
    for r in range(data_start, len(df)):
        if pd.isna(df.iloc[r, 0]):
            total_row = r
            break
    if total_row is None:
        continue

    # Define blocks using headers
    blocks = [
        {'khoi': 'A', 'subjects': [str(df.iloc[header_row, 1]), str(df.iloc[header_row, 2]), str(df.iloc[header_row, 3])], 'cols': [1, 2, 3]},
        {'khoi': 'A1', 'subjects': [str(df.iloc[header_row, 5]), str(df.iloc[header_row, 6]), str(df.iloc[header_row, 7])], 'cols': [5, 6, 7]},
        {'khoi': 'B', 'subjects': [str(df.iloc[header_row, 9]), str(df.iloc[header_row, 10]), str(df.iloc[header_row, 11])], 'cols': [9, 10, 11]},
        {'khoi': 'C', 'subjects': [str(df.iloc[header_row, 13]), str(df.iloc[header_row, 14]), str(df.iloc[header_row, 15])], 'cols': [13, 14, 15]},
        {'khoi': 'D', 'subjects': [str(df.iloc[header_row, 17]), str(df.iloc[header_row, 18]), str(df.iloc[header_row, 19])], 'cols': [17, 18, 19]},
    ]

    # Process each block
    for block in blocks:
        for sub_idx, subject in enumerate(block['subjects']):
            if pd.isna(subject) or subject == 'nan':
                continue
            col = block['cols'][sub_idx]
            total = df.iloc[total_row, col]
            if pd.isna(total) or total == 0:
                continue

            # Collect score to count
            score_to_count = {}
            for r in range(data_start, total_row):
                score = df.iloc[r, 0]
                count = df.iloc[r, col]
                if pd.notna(score) and pd.notna(count):
                    score_to_count[score] = count

            if not score_to_count:
                continue

            # Sort scores descending
            sorted_scores = sorted(score_to_count.keys(), reverse=True)

            # Compute cumulatives (>= score)
            cumuls = {}
            current_cumul = 0.0
            for s in sorted_scores:
                current_cumul += score_to_count[s]
                cumuls[s] = current_cumul

            # For each score, compute IQ15 using new method
            for s in sorted_scores:
                count = score_to_count[s]
                cumul = cumuls[s]
                if pd.isna(cumul) or pd.isna(count):
                    continue
                num_strictly_below = total - cumul
                proportion = num_strictly_below / total if total > 0 else 0
                if proportion > epsilon and proportion < (1.0 - epsilon):
                    z = norm.ppf(proportion)
                    iq_value = 100 + 15 * z
                    iq = f"{iq_value:.7f}".rstrip('0').rstrip('.')
                else:
                    iq = '-'
                output.append({
                    'Year': year,
                    'Subject': subject,
                    'khoi_thi': block['khoi'],
                    'Score': s,
                    'count': count,
                    'Cumulative': cumul,
                    'IQ15': iq
                })

# Create DataFrame and save to CSV
out_df = pd.DataFrame(output)
out_df.to_csv('transformed.csv', index=False)
print("CSV file 'transformed.csv' has been created.")