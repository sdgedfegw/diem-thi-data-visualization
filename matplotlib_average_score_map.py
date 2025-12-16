import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.table import Table
import numpy as np
import os
import textwrap 

# ==========================================
# 1. CONFIGURATION & MAPPINGS
# ==========================================

# File Paths
GEOJSON_PATH = 'Viet Nam_tinh thanh.geojson'
PROVINCES_CSV_PATH = 'vietnam_provinces.csv'
AVG_SCORES_CSV_PATH = 'average_scores_2016_2025.csv'
DIST_SCORES_CSV_PATH = 'score_distribution_provinces_2016_2025.csv'
OUTPUT_DIR = 'output_maps'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subject Name Mapping
SUBJECT_NAME_MAP = {
    "NguVan": "Ngữ văn", "Toan": "Toán", "NgoaiNgu": "Ngoại ngữ",
    "VatLy": "Vật lí", "HoaHoc": "Hóa học", "SinhHoc": "Sinh học",
    "LichSu": "Lịch sử", "DiaLy": "Địa lí", "GDCD": "Giáo dục công dân",
    "KHTN": "3 môn Khoa học tự nhiên", "KHXH": "3 môn Khoa học xã hội",
    "KhoiA": "A", "KhoiA1": "A1", "KhoiB": "B",
    "KhoiC": "C", "KhoiD": "D", "KhoiA02": "A02",
    "KhoiC01": "C01", "KhoiD07": "D07",
    "TongDiem": "tổng điểm thi THPT",
    "TongDiemKHTN": "tổng điểm thi THPT Tổ hợp Khoa học tự nhiên",
    "TongDiemKHXH": "tổng điểm thi THPT Tổ hợp Khoa học xã hội",
    # NEW SUBJECTS ADDED
    "CongNgheCongNghiep": "Công nghệ - Công nghiệp",
    "CongNgheNongNghiep": "Công nghệ - Nông nghiệp",
    "KinhTePhapLuat": "Kinh tế Pháp luật",
    "TinHoc": "Tin học"
}

# Custom Min/Max Limits for Color Scaling
SUBJECT_COLOR_LIMITS = {
    "DiaLy": (4.16, 8.07),
    "HoaHoc": (3.93, 7.52),
    "KHTN": (3.88, 7.29),
    "KhoiA": (12.2, 22.93),
    "KhoiA02": (12.22, 22.5),
    "KhoiA1": (11.41, 22.46),
    "KhoiB": (12.32, 22.52),
    "KhoiC": (12.19, 23.42),
    "KhoiC01": (13.08, 23.46),
    "KhoiD": (10.37, 21.71),
    "KhoiD07": (11.49, 22.5),
    "LichSu": (2.92, 7.44),
    "NgoaiNgu": (2.58, 7.23),
    "NguVan": (3.65, 8.17),
    "SinhHoc": (3.89, 7.32),
    "Toan": (2.99, 7.83),
    "TongDiem": (21.03, 45.88),
    "TongDiemKHTN": (24.26, 44.04),
    "VatLy": (3.81, 7.5),
    "GDCD": (5.98, 9.11),
    "KHXH": (4.74, 8.45),
    "TongDiemKHXH": (24.87, 46.79),
    # NEW SUBJECTS ADDED
    "CongNgheCongNghiep": (4.1, 8.62),
    "CongNgheNongNghiep": (7.03, 8.5),
    "KinhTePhapLuat": (6.82, 8.43),
    "TinHoc": (5.23, 9.11)
}

# Groupings
# Added new subjects to GROUP_MON
GROUP_MON = [
    "Toan", "NguVan", "VatLy", "HoaHoc", "SinhHoc", "LichSu", "DiaLy", 
    "GDCD", "NgoaiNgu", "KHTN", "KHXH", 
    "CongNgheCongNghiep", "CongNgheNongNghiep", "KinhTePhapLuat", "TinHoc"
]

GROUP_KHOI = ["KhoiA", "KhoiA1", "KhoiB", "KhoiC", "KhoiD", "KhoiC01", "KhoiD07"]
GROUP_TONG = ["TongDiem", "TongDiemKHTN", "TongDiemKHXH"]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def create_custom_colormap():
    """Creates a colormap that transitions from Red -> Orange -> Dark Yellow -> Green."""
    colors = [
        (0.0, "#d73027"), # Red
        (0.25, "#fc8d59"), # Orange
        (0.5,  "#CCCC00"), # Darker Yellow/Mustard
        (0.75, "#91cf60"), # Light Green
        (1.0,  "#1a9850")  # Dark Green
    ]
    return mcolors.LinearSegmentedColormap.from_list("custom_exam_cmap", colors)

def get_max_score_theoretical(subject, year):
    """Returns theoretical max score for Table Threshold calculation (15, 18, 24 etc)"""
    if subject in GROUP_MON:
        return 10.0
    elif subject in GROUP_KHOI:
        return 30.0
    elif subject in GROUP_TONG:
        if year <= 2024:
            return 60.0
        else:
            return 40.0
    return 10.0 

def get_chart_title(year, subject_code):
    subject_name_vn = SUBJECT_NAME_MAP.get(subject_code, subject_code)
    subject_upper = subject_name_vn.upper()
    
    # Check if subject is in the list of standard subjects including new ones
    is_standard_subject = subject_code in [
        "Toan", "NguVan", "VatLy", "HoaHoc", "SinhHoc", "LichSu", "DiaLy", 
        "GDCD", "NgoaiNgu", "CongNgheCongNghiep", "CongNgheNongNghiep", 
        "KinhTePhapLuat", "TinHoc"
    ]

    if year <= 2019:
        if is_standard_subject:
            return f"BẢN ĐỒ KẾT QUẢ THI THPT QUỐC GIA DỰA THEO ĐIỂM TRUNG BÌNH MÔN {subject_upper} NĂM {year}"
        elif subject_code in ["KHTN", "KHXH", "TongDiem", "TongDiemKHTN", "TongDiemKHXH"]:
            return f"BẢN ĐỒ KẾT QUẢ THI THPT QUỐC GIA DỰA THEO ĐIỂM TRUNG BÌNH {subject_upper} NĂM {year}"
        elif subject_code in GROUP_KHOI:
            return f"BẢN ĐỒ KẾT QUẢ THI THPT QUỐC GIA DỰA THEO ĐIỂM TRUNG BÌNH KHỐI {subject_upper} NĂM {year}"
    else:
        if is_standard_subject:
            return f"BẢN ĐỒ KẾT QUẢ THI TỐT NGHIỆP THPT DỰA THEO ĐIỂM TRUNG BÌNH MÔN {subject_upper} NĂM {year}"
        elif subject_code in ["KHTN", "KHXH", "TongDiem", "TongDiemKHTN", "TongDiemKHXH"]:
            return f"BẢN ĐỒ KẾT QUẢ THI TỐT NGHIỆP THPT DỰA THEO ĐIỂM TRUNG BÌNH {subject_upper} NĂM {year}"
        elif subject_code in GROUP_KHOI:
            return f"BẢN ĐỒ KẾT QUẢ THI TỐT NGHIỆP THPT DỰA THEO ĐIỂM TRUNG BÌNH KHỐI {subject_upper} NĂM {year}"
            
    return f"BẢN ĐỒ KẾT QUẢ THI NĂM {year} - {subject_upper}"

def standardize_province_code(val):
    try:
        s = str(val).split('.')[0]
        return s.zfill(2)
    except:
        return str(val).zfill(2)

# ==========================================
# 3. DATA PROCESSING
# ==========================================

def clean_data_frame(df_input, year_col='Year', prov_col='Province_Code', score_cols=[]):
    df = df_input.copy()
    df.columns = df.columns.str.strip()
    
    if year_col in df.columns:
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        df = df.dropna(subset=[year_col])
        df[year_col] = df[year_col].astype(int)
    
    if prov_col in df.columns:
        df[prov_col] = df[prov_col].apply(standardize_province_code)
    
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def load_and_prep_data():
    gdf = gpd.read_file(GEOJSON_PATH)
    
    df_prov = pd.read_csv(PROVINCES_CSV_PATH, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    df_prov.columns = df_prov.columns.str.strip()
    
    df_prov['ma_tinh'] = df_prov['ma_tinh'].apply(standardize_province_code)
    df_prov['Province_Code'] = df_prov['Province_Code'].apply(standardize_province_code)
    
    cols_to_merge = ['ma_tinh', 'Province_Code']
    if 'ten_tinh' not in gdf.columns:
        cols_to_merge.append('ten_tinh')
        
    gdf = gdf.merge(df_prov[cols_to_merge], on='ma_tinh', how='left')
    return gdf, df_prov

def get_stats_for_table(year, subject, max_score, df_dist, df_avg, df_prov):
    percentages = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresholds = [p * max_score for p in percentages]
    
    sub_dist = df_dist[(df_dist['Year'] == year) & (df_dist['Subject'] == subject)].copy()
    sub_avg = df_avg[(df_avg['Year'] == year) & (df_avg['Subject'] == subject)].copy()
    
    table_rows = []
    valid_provinces = df_prov['Province_Code'].unique()
    
    national_sums = {t: 0 for t in thresholds}
    national_total_count = 0
    
    for prov_code in valid_provinces:
        prov_name_series = df_prov[df_prov['Province_Code'] == prov_code]['ten_tinh']
        if prov_name_series.empty: continue
        prov_name = prov_name_series.values[0]
        
        avg_row = sub_avg[sub_avg['Province_Code'] == prov_code]
        avg_score = avg_row['Average_Score'].values[0] if not avg_row.empty else 0.0
        
        prov_dist = sub_dist[sub_dist['Province_Code'] == prov_code]
        
        if prov_dist.empty and avg_row.empty:
            continue
            
        if 'Cumulative' in prov_dist.columns:
            total_count = prov_dist['Cumulative'].max()
        else:
            total_count = prov_dist['Count'].sum()
            
        if pd.isna(total_count) or total_count == 0:
            total_count = prov_dist['Count'].sum()

        national_total_count += total_count
        
        row_data = {
            'Name': prov_name,
            'Count': int(total_count),
            'Avg': float(avg_score) if not pd.isna(avg_score) else 0.0
        }
        
        for t in thresholds:
            count_ge = prov_dist[prov_dist['Score'] >= t]['Count'].sum()
            row_data[f'ge_{t}'] = int(count_ge)
            national_sums[t] += int(count_ge)
            
        table_rows.append(row_data)
        
    table_rows.sort(key=lambda x: x['Avg'], reverse=True)
    
    for i, row in enumerate(table_rows):
        row['STT'] = i + 1
    
    nat_avg_row = sub_avg[sub_avg['Province_Code'].isin(['99', 'CaNuoc', '00'])]
    national_avg = 0.0
    if not nat_avg_row.empty:
        national_avg = nat_avg_row['Average_Score'].values[0]
    
    national_row = {
        'STT': '',
        'Name': 'Cả nước',
        'Count': int(national_total_count),
        'Avg': float(national_avg)
    }
    for t in thresholds:
        national_row[f'ge_{t}'] = national_sums[t]
        
    return table_rows, national_row, thresholds

# ==========================================
# 4. PLOTTING FUNCTION
# ==========================================

def generate_exam_map(year, subject):
    print(f"Processing: Year {year}, Subject {subject}")
    
    # 1. Load Data
    gdf, df_prov = load_and_prep_data()
    
    raw_avg = pd.read_csv(AVG_SCORES_CSV_PATH, dtype=str, low_memory=False, encoding='utf-8-sig')
    raw_dist = pd.read_csv(DIST_SCORES_CSV_PATH, dtype=str, low_memory=False, encoding='utf-8-sig')
    
    df_avg = clean_data_frame(raw_avg, year_col='Year', prov_col='Province_Code', score_cols=['Average_Score'])
    df_dist = clean_data_frame(raw_dist, year_col='Year', prov_col='Province_Code', score_cols=['Score', 'Count', 'Cumulative'])
    
    # 2. Settings
    # Theoretical Max for Table Calculations (15, 18, 24...)
    theoretical_max = get_max_score_theoretical(subject, year)
    
    # Custom Min/Max for Color Scaling
    if subject in SUBJECT_COLOR_LIMITS:
        vmin, vmax = SUBJECT_COLOR_LIMITS[subject]
    else:
        # Fallback if subject not in list
        vmin, vmax = 0, theoretical_max
        
    title_text = get_chart_title(year, subject)
    custom_cmap = create_custom_colormap()
    
    # 3. Prepare Map Data
    current_avg = df_avg[(df_avg['Year'] == year) & (df_avg['Subject'] == subject)].copy()
    map_data = gdf.merge(current_avg, on='Province_Code', how='left')
    
    # 4. Prepare Table Data
    rows, nat_row, thresholds = get_stats_for_table(year, subject, theoretical_max, df_dist, df_avg, df_prov)
    
    # 5. Initialize Plot
    plt.rcParams['font.family'] = 'Times New Roman'
    fig = plt.figure(figsize=(50, 50), dpi=100)
    
    ax_map = fig.add_axes([0, 0, 1, 1]) 
    ax_table = fig.add_axes([0.65, 0.15, 0.30, 0.60]) 
    # Title moved higher: Bottom 0.94, Height 0.05
    ax_title = fig.add_axes([0.05, 0.98, 0.9, 0.05]) 
    
    # 6. Draw Title with Wrapping
    ax_title.axis('off')
    
    # Wrap text at approx 65 chars (Fits ~90% of width at font size 80)
    wrapped_title = textwrap.fill(title_text.upper(), width=65)
    
    ax_title.text(0.5, 0.5, wrapped_title, ha='center', va='center', 
                  fontsize=80, fontweight='bold', color='#1a237e')
    
    # 7. Draw Map
    ax_map.axis('off')
    
    gdf.plot(ax=ax_map, color='#eeeeee', edgecolor='white', linewidth=0.8)
    
    valid_map_data = map_data.dropna(subset=['Average_Score'])
    
    if not valid_map_data.empty:
        # Apply specific vmin/vmax from SUBJECT_COLOR_LIMITS
        valid_map_data.plot(column='Average_Score', ax=ax_map, cmap=custom_cmap, 
                            vmin=vmin, vmax=vmax, edgecolor='white', linewidth=0.8)
        
        # Labels
        for idx, row in valid_map_data.iterrows():
            pt = row['geometry'].representative_point()
            x, y = pt.x, pt.y
            
            prov_name = ""
            if 'ten_tinh' in row and pd.notna(row['ten_tinh']):
                prov_name = row['ten_tinh']
            elif 'ten_tinh_x' in row and pd.notna(row['ten_tinh_x']):
                prov_name = row['ten_tinh_x']
            
            score_val = row['Average_Score']
            label_text = f"{prov_name}\n{score_val:.2f}"
            
            txt = ax_map.text(x, y, label_text, ha='center', va='center', fontsize=18, fontweight='bold', color='black')
            txt.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])
    else:
        print("Warning: No data for map.")
    
    # Legend with specific Limits
    cax = fig.add_axes([0.05, 0.05, 0.2, 0.02])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label('Thang điểm trung bình', size=35)

    # 8. Draw Table
    ax_table.axis('off')
    
    headers = ["STT", "Tên tỉnh/TP", "SL TS", "Điểm TB"] + [f"≥ {t:g}" for t in thresholds]
    
    cell_text = []
    fmt_int = lambda x: f"{x:,.0f}"
    fmt_float = lambda x: f"{x:.2f}"
    
    for r in rows:
        row_content = [str(r['STT']), r['Name'], fmt_int(r['Count']), fmt_float(r['Avg'])]
        for t in thresholds:
            row_content.append(fmt_int(r[f'ge_{t}']))
        cell_text.append(row_content)
        
    nat_content = ["", nat_row['Name'], fmt_int(nat_row['Count']), fmt_float(nat_row['Avg'])]
    for t in thresholds:
        nat_content.append(fmt_int(nat_row[f'ge_{t}']))
    cell_text.append(nat_content)
    
    the_table = Table(ax_table, bbox=[0, 0, 1, 1])
    
    n_thresholds = len(thresholds)
    
    # UPDATED: Fixed widths logic
    # [STT, Name, SL TS, Avg] = [0.05, 0.15, 0.08, 0.08]
    # w_rest (Threshold columns) = same width as "SL TS" = 0.08
    col_widths = [0.05, 0.15, 0.08, 0.08] + [0.08] * n_thresholds
    
    # Add Header
    for i, h in enumerate(headers):
        cell = the_table.add_cell(0, i, width=col_widths[i], height=0.03, 
                                  text=h, loc='center', facecolor='#e0e0e0')
        cell.get_text().set_weight('bold')
        # UPDATED: Font size increased by ~20% (28 -> 34)
        cell.get_text().set_fontsize(34)
    
    # Add Rows
    for row_idx, row_data in enumerate(cell_text):
        is_last = (row_idx == len(cell_text) - 1)
        row_color = '#fff3e0' if is_last else ('#ffffff' if row_idx % 2 == 0 else '#f9f9f9')
        font_weight = 'bold' if is_last else 'normal'
        
        for col_idx, val in enumerate(row_data):
            cell = the_table.add_cell(row_idx + 1, col_idx, width=col_widths[col_idx], height=0.022,
                                      text=str(val), loc='center', facecolor=row_color)
            # UPDATED: Font size increased by ~20% (24 -> 29)
            cell.get_text().set_fontsize(29)
            cell.get_text().set_weight(font_weight)
            if col_idx == 0: cell.get_text().set_weight('bold')

    ax_table.add_table(the_table)
    title_obj = ax_table.text(0.5, 1.01, "DỮ LIỆU CHI TIẾT", ha='center', va='bottom', fontsize=40, fontweight='bold', color='black')
    title_obj.set_path_effects([pe.withStroke(linewidth=4, foreground='white')])

    # 9. Save
    output_filename = f"{OUTPUT_DIR}/Map_{year}_{subject}.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=1)
    print(f"Success: {output_filename}")
    plt.close()

if __name__ == "__main__":
    # Load the data to identify available years
    df_all = pd.read_csv(AVG_SCORES_CSV_PATH)
    years = sorted(df_all['Year'].unique())
    
    target_subject = "KhoiD"

    print(f"Starting processing for Subject: {target_subject}")
    print(f"Years found: {years}")

    for year in years:
        try:
            # Check if data exists for this specific combination before plotting
            mask = (df_all['Year'] == year) & (df_all['Subject'] == target_subject)
            
            if not df_all[mask].empty:
                generate_exam_map(int(year), target_subject)
            else:
                print(f"Skipping: No data for {year} - {target_subject}")
                
        except Exception as e:
            print(f"Error processing {year} - {target_subject}: {e}")
            
    print("Processing complete!")