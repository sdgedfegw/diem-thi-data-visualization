import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import numpy as np
import os
import math

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

# --- DATA MAPPING ---

SUBJECT_NAME_MAP = {
    "NguVan": "Ngữ văn",
    "Toan": "Toán",
    "NgoaiNgu": "Ngoại ngữ",
    "VatLy": "Vật lí",
    "HoaHoc": "Hóa học",
    "SinhHoc": "Sinh học",
    "LichSu": "Lịch sử",
    "DiaLy": "Địa lí",
    "GDCD": "Giáo dục công dân",
    "KinhTePhapLuat": "Kinh tế Pháp luật",
    "TinHoc": "Tin học",
    "CongNgheCongNghiep": "Công nghệ - Công nghiệp",
    "CongNgheNongNghiep": "Công nghệ - Nông nghiệp"
}

# Step configuration table
# Structure: {Year: {Subject: Step}}
STEP_CONFIG = {
    2025: {"default": 0.25, "GDCD": None}, 
    2024: {"Toan": 0.2, "NgoaiNgu": 0.2, "default": 0.25},
    2023: {"Toan": 0.2, "NgoaiNgu": 0.2, "default": 0.25},
    2022: {"Toan": 0.2, "NgoaiNgu": 0.2, "default": 0.25},
    2021: {"Toan": 0.2, "NgoaiNgu": 0.2, "default": 0.25},
    2020: {"Toan": 0.2, "NgoaiNgu": 0.2, "default": 0.25},
    2019: {"Toan": 0.2, "NgoaiNgu": 0.2, "default": 0.25},
    2018: {"Toan": 0.2, "NgoaiNgu": 0.2, "default": 0.25},
    2017: {"Toan": 0.2, "NgoaiNgu": 0.2, "default": 0.25},
    2016: {"Toan": 0.25, "NgoaiNgu": 0.2, "VatLy": 0.2, "HoaHoc": 0.2, "SinhHoc": 0.2, "default": 0.25},
}
# Years 2007-2015 are all 0.25 according to the requirements table
for y in range(2007, 2016):
    STEP_CONFIG[y] = {"default": 0.25}

def get_step_size(year, subject):
    year_int = int(year)
    if year_int not in STEP_CONFIG:
        return 0.25
    config = STEP_CONFIG[year_int]
    if subject in config:
        val = config[subject]
        if val is None: return None
        return val
    if "default" in config:
        return config["default"]
    return 0.25

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
    while (y_max / candidates[current_idx]) > 10 and current_idx < len(candidates) - 1:
        current_idx += 1
    while (y_max / candidates[current_idx]) < 4 and current_idx > 0:
        current_idx -= 1
    return candidates[current_idx]

def create_custom_colormap():
    colors = [
        (0.0, "#d73027"), # Red
        (0.25, "#fc8d59"), # Orange
        (0.5,  "#CCCC00"), # Darker Yellow/Mustard
        (0.75, "#91cf60"), # Light Green
        (1.0,  "#1a9850")  # Dark Green
    ]
    return mcolors.LinearSegmentedColormap.from_list("custom_exam_cmap", colors)

def find_score_at_percentile(df, target_cumulative_count):
    temp_df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    temp_df['cum_check'] = temp_df['count'].cumsum()
    idx = (temp_df['cum_check'] - target_cumulative_count).abs().idxmin()
    return temp_df.loc[idx, 'score']

def process_data_binning(df, step):
    df_proc = df.copy()
    df_proc['Score'] = pd.to_numeric(df_proc['Score'], errors='coerce')
    df_proc['count'] = pd.to_numeric(df_proc['count'], errors='coerce').fillna(0)
    
    def calculate_bin(score):
        if pd.isna(score): return 0
        if math.isclose(score, 10.0, rel_tol=1e-9):
            return 10.0
        val = score / step
        floored = math.floor(round(val, 6))
        return round(floored * step, 3)

    df_proc['score_bin'] = df_proc['Score'].apply(calculate_bin)
    grouped = df_proc.groupby('score_bin')['count'].sum().reset_index()
    grouped.rename(columns={'score_bin': 'score'}, inplace=True)
    return grouped

def generate_chart(data_df, year, subject, khoi_label, step):
    # 1. Process Data
    grouped_df = process_data_binning(data_df, step)
    
    num_steps = int(10.0 / step) + 1
    all_scores = np.linspace(0, 10, num_steps)
    all_scores = np.round(all_scores, 3)
    
    df_grid = pd.DataFrame({'score': all_scores})
    df_merged = pd.merge(df_grid, grouped_df, on='score', how='left')
    df_merged['count'] = df_merged['count'].fillna(0)
    df_merged = df_merged.sort_values(by='score', ascending=True)

    x = df_merged['score'].values
    y = df_merged['count'].values
    
    max_count = y.max()
    if max_count <= 0:
        print(f"Skipping {year} {subject}: No data.")
        return

    # Integer casting for natural number display
    total_candidates = int(df_merged['count'].sum())
    
    # 2. Statistics
    weighted_sum = np.sum(x * y)
    avg_score = weighted_sum / total_candidates if total_candidates > 0 else 0
    
    max_score_row = df_merged[df_merged['count'] > 0].iloc[-1]
    max_score_val = max_score_row['score']
    max_score_count = int(max_score_row['count'])
    
    highest_score_str = f"Điểm cao nhất: {max_score_val:g} ({max_score_count} thí sinh)\n\n"
    
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
    df_cum = df_merged.copy()
    for z, label_pct, prob in z_defs:
        target_cum = total_candidates * (1.0 - prob)
        val = find_score_at_percentile(df_cum, target_cum)
        sign = "+" if z > 0 else ""
        z_stats_text += f"  Độ lệch chuẩn {sign}{z} (Top {label_pct}%): {val:g}\n"

    def get_count_stats(threshold, is_exact=False):
        if is_exact:
            cnt = df_merged[np.isclose(df_merged['score'], threshold)]['count'].sum()
        else:
            cnt = df_merged[df_merged['score'] >= (threshold - 0.001)]['count'].sum()
        pct = (100 - cnt / total_candidates * 100) if total_candidates > 0 else 0
        return cnt, pct

    c5, p5 = get_count_stats(5)
    c6, p6 = get_count_stats(6)
    c7, p7 = get_count_stats(7)
    c8, p8 = get_count_stats(8)
    c9, p9 = get_count_stats(9)
    c10, p10 = get_count_stats(10, is_exact=True)

    # 3. Setup Figure
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    y_axis_max = max_count * 4 / 3
    ax.set_ylim(0, y_axis_max)
    ax.set_xlim(-0.1, 10.1)
    
    # 4. Color Gradient
    cmap = create_custom_colormap()
    norm = mcolors.Normalize(vmin=0, vmax=10)
    colors = cmap(norm(x))
    
    # 5. Plot Bars
    rects = ax.bar(x, y, width=step*0.8, color=colors, align='center', zorder=3)
    
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
    y_step = get_y_tick_step(y_axis_max)
    y_ticks = np.arange(0, y_axis_max + (y_step*0.1), y_step)
    y_ticks = y_ticks[y_ticks <= y_axis_max * 1.05]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # 7. X-Axis
    ax.set_xticks(all_scores)
    xtick_labels = [f"{v:g}" for v in all_scores]
    rotation = 90
    font_size_x = 9 * SCALE_H
    if step < 0.25: font_size_x = 8 * SCALE_H
    ax.set_xticklabels(xtick_labels, rotation=rotation, fontsize=font_size_x)

    # 8. Styling
    ax.grid(True, which='major', axis='both', linestyle='-', linewidth=0.5 * SCALE_W, alpha=0.3, color='#555555', zorder=0)
    
    title_fs = 32 * SCALE_H
    label_fs = 20 * SCALE_H
    tick_fs = 14 * SCALE_H
    legend_fs = 12 * SCALE_H 
    
    ax.tick_params(axis='y', labelsize=tick_fs)

    # Title
    year_int = int(year)
    subject_vn = SUBJECT_NAME_MAP.get(subject, subject)
    if year_int <= 2014:
        title_text = f"Biểu đồ phổ điểm thi Đại học môn {subject_vn} - Khối {khoi_label} - Năm {year}"
    elif 2015 <= year_int <= 2019:
        title_text = f"Biểu đồ phổ điểm thi THPT Quốc gia môn {subject_vn} năm {year}"
    else:
        title_text = f"Biểu đồ phổ điểm thi Tốt nghiệp THPT môn {subject_vn} năm {year}"

    plt.title(title_text, fontsize=title_fs, fontweight='bold', pad=35 * SCALE_H)
    ax.set_xlabel("Điểm số", fontsize=label_fs, labelpad=25 * SCALE_H)
    ax.set_ylabel("Số lượng thí sinh", fontsize=label_fs, labelpad=35 * SCALE_H)

    # Top Left Text (Total Candidates) - Formatted as Int
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
        f"  Điểm ≥ 5: {int(c5):,} (Top {p5:.2f}%)\n"
        f"  Điểm ≥ 6: {int(c6):,} (Top {p6:.2f}%)\n"
        f"  Điểm ≥ 7: {int(c7):,} (Top {p7:.2f}%)\n"
        f"  Điểm ≥ 8: {int(c8):,} (Top {p8:.2f}%)\n"
        f"  Điểm ≥ 9: {int(c9):,} (Top {p9:.2f}%)\n"
        f"  Điểm = 10: {int(c10):,} (Top {p10:.2f}%)"
    )
    
    props = dict(boxstyle='square,pad=1', facecolor='white', alpha=0.75, edgecolor='black', linewidth=2 * SCALE_W)
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=legend_fs,
            verticalalignment='top', horizontalalignment='left', bbox=props, zorder=5)

    # 10. Save
    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.96)
    
    if year_int <= 2014:
        filename_base = f"score_dist_mon_{year}_{subject}_{khoi_label}"
    else:
        filename_base = f"score_dist_mon_{year}_{subject}"

    print(f"Saving {filename_base}...")
    plt.savefig(f"{filename_base}.svg", format='svg')
    plt.savefig(f"{filename_base}.png", format='png', dpi=DPI)
    plt.close(fig)

def main():
    input_csv = 'matplotlib_score_dist_preprocess_mon.csv'
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)

    unique_years = sorted(df['Year'].unique())

    for year in unique_years:
        year_df = df[df['Year'] == year]
        subjects = year_df['Subject'].unique()
        
        for subject in subjects:
            step = get_step_size(year, subject)
            if step is None: continue
            
            subject_df = year_df[year_df['Subject'] == subject]
            
            if year <= 2014:
                unique_khois = subject_df['khoi'].unique()
                for khoi in unique_khois:
                    if pd.isna(khoi): continue
                    data_subset = subject_df[subject_df['khoi'] == khoi]
                    generate_chart(data_subset, year, subject, khoi, step)
            else:
                generate_chart(subject_df, year, subject, "", step)

    print("Processing complete.")

if __name__ == "__main__":
    main()