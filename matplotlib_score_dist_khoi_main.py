import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import numpy as np
import os

# Use Agg backend for non-interactive image generation
plt.switch_backend('Agg')

# --- CONFIGURATION ---
IMG_WIDTH_PX = 5000
IMG_HEIGHT_PX = 2813
DPI = 100
FIG_SIZE = (IMG_WIDTH_PX / DPI, IMG_HEIGHT_PX / DPI)

# Scaling factors
SCALE_H = IMG_HEIGHT_PX / 1000 
SCALE_W = IMG_WIDTH_PX / 1000

# Font Configuration: Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 

def get_y_tick_step(y_max):
    """Calculates the Y-axis tick step (4-10 labels constraint)."""
    if y_max <= 0: return 1.0
    target = y_max / 10.0
    exponent = np.floor(np.log10(target)) if target > 0 else 0
    magnitude = 10 ** exponent
    multipliers = [1, 2, 5]
    candidates = []
    for p in [magnitude/10, magnitude, magnitude*10]:
        for m in multipliers:
            candidates.append(m * p)
    candidates = sorted(list(set(candidates)))
    best_step = min(candidates, key=lambda x: abs(x - target))
    try:
        current_idx = candidates.index(best_step)
    except ValueError:
        current_idx = 0
    # Adjust to fit constraints
    while (y_max / candidates[current_idx]) > 10 and current_idx < len(candidates) - 1:
        current_idx += 1
    while (y_max / candidates[current_idx]) < 4 and current_idx > 0:
        current_idx -= 1
    return candidates[current_idx]

def create_custom_colormap():
    """
    Creates a colormap that transitions from Red -> Orange -> Dark Yellow -> Green.
    """
    colors = [
        (0.0, "#d73027"), # Red
        (0.25, "#fc8d59"), # Orange
        (0.5,  "#CCCC00"), # Darker Yellow/Mustard
        (0.75, "#91cf60"), # Light Green
        (1.0,  "#1a9850")  # Dark Green
    ]
    return mcolors.LinearSegmentedColormap.from_list("custom_exam_cmap", colors)

def find_score_at_percentile(df, target_cumulative_count):
    """
    Finds the score where the cumulative count matches the target.
    """
    idx = (df['cumulative'] - target_cumulative_count).abs().idxmin()
    return df.loc[idx, 'min_score']

def generate_chart(group_df, year, khoi, high_score_data=None):
    # 1. Prepare Data
    all_scores = np.arange(0, 30.25, 0.25)
    df_grid = pd.DataFrame({'min_score': all_scores})
    df_merged = pd.merge(df_grid, group_df, on='min_score', how='left')
    df_merged['count'] = df_merged['count'].fillna(0)
    
    # Calculate cumulative
    df_merged = df_merged.sort_values(by='min_score', ascending=False).reset_index(drop=True)
    df_merged['cumulative'] = df_merged['count'].cumsum()
    
    # Sort back to ascending for plotting
    df_merged = df_merged.sort_values(by='min_score', ascending=True)

    x = df_merged['min_score'].values
    y = df_merged['count'].values
    
    max_count = y.max()
    if max_count <= 0:
        print(f"Skipping Year {year} Khoi {khoi}: No data.")
        return

    total_candidates = df_merged['count'].sum()
    
    # 2. Statistics Calculation
    # Mean
    weighted_sum = np.sum(x * y)
    avg_score = weighted_sum / total_candidates if total_candidates > 0 else 0
    
    # Prepare Highest Score String
    highest_score_str = ""
    if high_score_data is not None and not high_score_data.empty:
        # Extract the first row match (should be unique per year/khoi)
        h_score = high_score_data.iloc[0]['highest_score']
        h_count = high_score_data.iloc[0]['so_luong']
        # Convert count to int to remove decimals if any
        highest_score_str = f"Điểm cao nhất: {h_score:.2f} ({int(h_count)} thí sinh)\n\n"
    
    # Percentile / Z-Score Logic
    z_defs = [
        (3,  "99.87", 0.9987),
        (2,  "97.72", 0.9772),
        (1,  "84.13", 0.8413),
        (0,  "50",    0.5000),
        (-1, "15.87", 0.1587),
        (-2, "2.28",  0.0228),
        (-3, "0.13",  0.0013)
    ]
    
    z_stats_text = ""
    for z, label_pct, prob in z_defs:
        target_cum = total_candidates * (1.0 - prob)
        val = find_score_at_percentile(df_merged, target_cum)
        sign = "+" if z > 0 else ""
        z_stats_text += f"  Độ lệch chuẩn {sign}{z} (Top {label_pct}%): {val:.2f}\n"

    # Specific Counts & Percentages
    def get_count_stats(threshold, is_exact=False):
        if is_exact:
            cnt = df_merged[df_merged['min_score'] == threshold]['count'].sum()
        else:
            cnt = df_merged[df_merged['min_score'] >= threshold]['count'].sum()
        
        pct = (100 - cnt / total_candidates * 100) if total_candidates > 0 else 0
        return cnt, pct

    c15, p15 = get_count_stats(15)
    c18, p18 = get_count_stats(18)
    c21, p21 = get_count_stats(21)
    c24, p24 = get_count_stats(24)
    c27, p27 = get_count_stats(27)
    c30, p30 = get_count_stats(30, is_exact=True)

    # 3. Setup Figure
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    y_axis_max = max_count * 4 / 3
    ax.set_ylim(0, y_axis_max)
    ax.set_xlim(-0.1, 30.1)
    
    # 4. Color Gradient
    cmap = create_custom_colormap()
    norm = mcolors.Normalize(vmin=0, vmax=30)
    colors = cmap(norm(x))
    
    # 5. Plot Bars
    # Captured 'rects' to iterate over them for labeling
    rects = ax.bar(x, y, width=0.2, color=colors, align='center', zorder=3)
    
    # --- ADD LABELS TO BARS ---
    # Heuristic: If bar height > 5% of Y-axis, put inside. Else, outside.
    label_threshold = y_axis_max * 0.05
    label_font_size = 9 * SCALE_H
    
    for rect, val in zip(rects, y):
        if val == 0: continue
        
        height = rect.get_height()
        label_str = f"{int(val):,}" # Round to natural number
        
        if height > label_threshold:
            # Inside bar -> Bottom of the bar
            # Add small padding (0.5% of max height) so it doesn't touch the x-axis line
            y_pos = y_axis_max * 0.005 
            va = 'bottom'
            text_color = 'black' 
        else:
            # Outside bar -> Just above the bar
            y_pos = height + (y_axis_max * 0.005)
            va = 'bottom'
            text_color = 'black'
            
        ax.text(
            rect.get_x() + rect.get_width() / 2, 
            y_pos, 
            label_str, 
            ha='center', 
            va=va, 
            rotation=90, # Rotate to fit
            fontsize=label_font_size,
            color=text_color,
            zorder=4
        )

    # 6. Y-Axis
    step = get_y_tick_step(y_axis_max)
    y_ticks = np.arange(0, y_axis_max + (step*0.1), step)
    y_ticks = y_ticks[y_ticks <= y_axis_max * 1.05]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # 7. X-Axis
    ax.set_xticks(all_scores)
    xtick_labels = [f"{v:g}" for v in all_scores]
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=9 * SCALE_H)

    # 8. Styling & Layout
    ax.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5 * SCALE_W, alpha=0.6, color='#555555', zorder=0)
    
    title_fs = 32 * SCALE_H
    label_fs = 20 * SCALE_H
    tick_fs = 14 * SCALE_H
    legend_fs = 12 * SCALE_H 
    
    ax.tick_params(axis='y', labelsize=tick_fs)

    # Dynamic Title
    year_int = int(year)
    if year_int <= 2014:
        title_text = f"Biểu đồ phổ điểm thi Đại học khối {khoi} năm {year}"
    elif 2015 <= year_int <= 2019:
        title_text = f"Biểu đồ phổ điểm thi THPT Quốc gia khối {khoi} năm {year}"
    else: # >= 2020
        title_text = f"Biểu đồ phổ điểm thi Tốt nghiệp THPT khối {khoi} năm {year}"

    plt.title(title_text, fontsize=title_fs, fontweight='bold', pad=35 * SCALE_H)

    ax.set_xlabel("Khoảng điểm", fontsize=label_fs, labelpad=25 * SCALE_H)
    ax.set_ylabel("Số lượng thí sinh", fontsize=label_fs, labelpad=35 * SCALE_H)

    # Top Left Text (Total Candidates)
    ax.text(0, 1.01, f"Số lượng thí sinh: {total_candidates:,}", 
            transform=ax.transAxes, fontsize=label_fs, fontweight='bold', va='bottom', ha='left')

    # 9. Legend Box
    legend_text = (
        f"Các tham số đặc trưng:\n"
        f"─────────────────────\n"
        f"Điểm trung bình: {avg_score:.2f}\n\n"
        f"{highest_score_str}"
        f"{z_stats_text}\n"
        f"Số lượng thí sinh:\n"
        f"  Điểm ≥ 15: {int(c15):,} (Top {p15:.2f}%)\n"
        f"  Điểm ≥ 18: {int(c18):,} (Top {p18:.2f}%)\n"
        f"  Điểm ≥ 21: {int(c21):,} (Top {p21:.2f}%)\n"
        f"  Điểm ≥ 24: {int(c24):,} (Top {p24:.2f}%)\n"
        f"  Điểm ≥ 27: {int(c27):,} (Top {p27:.2f}%)\n"
        f"  Điểm = 30: {int(c30):,} (Top {p30:.2f}%)"
    )
    
    props = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='black', linewidth=2 * SCALE_W)
    
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=legend_fs,
            verticalalignment='top', horizontalalignment='left', bbox=props, zorder=5)

    # 10. Adjust Margins
    plt.subplots_adjust(
        top=0.90,     
        bottom=0.12,  
        left=0.08, 
        right=0.96
    )

    # 11. Save
    filename_base = f"score_dist_{year}_{khoi}"
    print(f"Saving {filename_base}...")
    plt.savefig(f"{filename_base}.svg", format='svg')
    plt.savefig(f"{filename_base}.png", format='png', dpi=DPI)
    plt.close(fig)

def main():
    input_csv = 'matplotlib_score_dist_preprocess_khoi.csv'
    highest_score_csv = 'highest_score.csv'
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    # Load Distribution Data
    df = pd.read_csv(input_csv)
    
    # Load Highest Score Data
    if os.path.exists(highest_score_csv):
        df_high = pd.read_csv(highest_score_csv)
    else:
        print(f"Warning: {highest_score_csv} not found. Charts will miss highest score info.")
        df_high = pd.DataFrame(columns=['year', 'khoi', 'highest_score', 'so_luong'])

    combinations = df[['year', 'khoi']].drop_duplicates().values

    for year, khoi in combinations:
        # Get distribution data
        group_data = df[(df['year'] == year) & (df['khoi'] == khoi)].copy()
        
        # Get highest score data for this specific year and khoi
        high_score_row = df_high[(df_high['year'] == year) & (df_high['khoi'] == khoi)]

        # Numeric conversion
        group_data['count'] = pd.to_numeric(group_data['count'], errors='coerce').fillna(0)
        
        generate_chart(group_data, year, khoi, high_score_row)
        
    print("Processing complete.")

if __name__ == "__main__":
    main()