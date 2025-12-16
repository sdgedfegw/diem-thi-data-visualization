import json
import csv
import os

# Configuration
input_filename = 'Viet Nam_tinh thanh.geojson'
output_filename = 'vietnam_provinces.csv'

def extract_to_csv():
    # Check if file exists
    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' was not found.")
        return

    print("Reading GeoJSON file...")
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Please check the file format.")
        return

    # List to hold unique records
    extracted_data = []
    # Set to track unique IDs to avoid duplicates (e.g. if geometry is split)
    seen_ids = set()

    # Iterate through features
    if 'features' in data:
        for feature in data['features']:
            props = feature.get('properties', {})
            
            # Get the unique identifier (ma_tinh)
            ma_tinh = props.get('ma_tinh')

            # Only add if we haven't seen this ID before
            if ma_tinh and ma_tinh not in seen_ids:
                record = {
                    'ma_tinh': ma_tinh,
                    'ten_tinh': props.get('ten_tinh'),
                    'loai': props.get('loai'),
                    'cap': props.get('cap'),
                    'stt': props.get('stt')
                }
                extracted_data.append(record)
                seen_ids.add(ma_tinh)
    
    # Sort data by 'stt' (optional, but makes the CSV cleaner)
    # handling None values just in case
    extracted_data.sort(key=lambda x: x['stt'] if x['stt'] is not None else 0)

    print(f"Extracted {len(extracted_data)} unique records.")
    print("Writing to CSV...")

    # Write to CSV
    # encoding='utf-8-sig' ensures Vietnamese characters show correctly in Excel
    with open(output_filename, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['ma_tinh', 'ten_tinh', 'loai', 'cap', 'stt']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(extracted_data)

    print(f"Done! Data saved to '{output_filename}'.")

if __name__ == "__main__":
    extract_to_csv()