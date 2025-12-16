import os

def bulk_rename_scores(directory_path):
    # Verify the directory exists
    if not os.path.exists(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.")
        return

    count = 0
    
    # Iterate through files in the specified folder
    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            
            # Separate the file name from the extension
            name_base, ext = os.path.splitext(filename)
            
            # Split the name by underscore
            parts = name_base.split('_')
            
            # Check structure: score_dist_mon_{subject}_{year}
            # Expected parts length is 5: ['score', 'dist', 'mon', 'subject', 'year']
            if len(parts) == 5 and parts[0] == 'score' and parts[1] == 'dist' and parts[2] == 'mon':
                
                # Extract variables based on input position
                subject = parts[3]
                year = parts[4]
                
                # Construct the new filename: score_dist_mon_{year}_{subject}
                new_filename = f"score_dist_mon_{year}_{subject}{ext}"
                
                # Define full file paths
                old_file_path = os.path.join(directory_path, filename)
                new_file_path = os.path.join(directory_path, new_filename)
                
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                    count += 1
                except Exception as e:
                    print(f"Error renaming {filename}: {e}")

    print(f"Processing complete. {count} files renamed.")

if __name__ == "__main__":
    # REPLACE THIS PATH with the actual path to your folder
    target_folder = r"D:\files\vietnam\phổ điểm thi matplotlib\mon_watermarked"
    
    bulk_rename_scores(target_folder)