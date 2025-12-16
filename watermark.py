import os
import math
from PIL import Image, ImageDraw, ImageFont, ImageColor

# ==========================================
#              CONFIGURATION
# ==========================================

# Folder paths ('.' means the current directory)
INPUT_FOLDER = r'D:\files\vietnam\phổ điểm thi matplotlib\output_maps'   # Folder containing your original PNGs
OUTPUT_FOLDER = r'D:\files\vietnam\phổ điểm thi matplotlib\output_maps_watermarked' # Folder where watermarked images will 

# Watermark Content
WATERMARK_TEXT = "tiktok.com/@diemthivn"

# Appearance
# Opacity: 0.0 is invisible, 1.0 is solid. 20% = 0.2
OPACITY = .7
TEXT_COLOR_HEX = "#BFBFBF"
IS_BOLD = True  # Used to select font file variant if available

# Font Settings
# NOTE: You must provide the path to the .ttf file on your system.
# Windows usually: "C:/Windows/Fonts/timesbd.ttf" (Times Bold)
# Mac usually: "/Library/Fonts/Times New Roman Bold.ttf"
# Linux: "/usr/share/fonts/..."
# If the script fails to load font, it will use a default ugly font.
FONT_PATH = "timesbd.ttf" # "times.ttf" for normal, "timesbd.ttf" for bold on Windows

# Sizing & Position
# 0.60 means the text width will match 60% of the image width
WIDTH_OCCUPANCY = 0.60

# ==========================================
#              END CONFIGURATION
# ==========================================

def add_watermark():
    # 1. Prepare Output Directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 2. Parse Color
    rgb = ImageColor.getrgb(TEXT_COLOR_HEX)
    # Combine RGB with Alpha (Opacity * 255)
    rgba_color = (rgb[0], rgb[1], rgb[2], int(255 * OPACITY))

    # 3. Process Files
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.png')]
    
    if not files:
        print(f"No PNG files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(files)} images. Starting processing...")

    for filename in files:
        try:
            full_path = os.path.join(INPUT_FOLDER, filename)
            
            # Open Image and ensure it is RGBA (for transparency support)
            with Image.open(full_path) as img:
                img = img.convert("RGBA")
                img_w, img_h = img.size

                # --- FONT SIZING LOGIC ---
                # We need to find a font size where the text width = 60% of image width
                target_text_width = img_w * WIDTH_OCCUPANCY
                
                font_size = 10  # Starting guess
                step_size = 10
                
                # Load font helper
                try:
                    font = ImageFont.truetype(FONT_PATH, font_size)
                except OSError:
                    print(f"Warning: Could not load {FONT_PATH}. Using default font.")
                    font = ImageFont.load_default()
                    # Default font usually doesn't scale well, but prevents crash

                # Iteratively find the right font size
                # (Binary search or simple loop - simple loop is fast enough for this)
                while True:
                    bbox = font.getbbox(WATERMARK_TEXT)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    
                    if text_w >= target_text_width:
                        break
                    
                    font_size += 5
                    try:
                        font = ImageFont.truetype(FONT_PATH, font_size)
                    except:
                        break # Keep previous size if scaling fails

                # --- ROTATION LOGIC ---
                # Calculate angle based on aspect ratio (diagonal)
                # Math.atan2(y, x) gives radians. Convert to degrees.
                # We rotate positive (counter-clockwise) to go bottom-left to top-right
                angle = math.degrees(math.atan2(img_h, img_w))

                # --- DRAWING ---
                # Create a separate transparent layer to draw text
                # We make it larger than the original image so rotation doesn't clip corners
                overlay_size = int(math.sqrt(img_w**2 + img_h**2)) + 50
                txt_layer = Image.new('RGBA', (overlay_size, overlay_size), (255, 255, 255, 0))
                draw = ImageDraw.Draw(txt_layer)

                # Get final text dimensions to center it on the overlay
                bbox = font.getbbox(WATERMARK_TEXT)
                final_text_w = bbox[2] - bbox[0]
                final_text_h = bbox[3] - bbox[1]

                # Draw text in the absolute center of the overlay
                text_x = (overlay_size - final_text_w) / 2
                text_y = (overlay_size - final_text_h) / 2
                
                draw.text((text_x, text_y), WATERMARK_TEXT, font=font, fill=rgba_color)

                # Rotate the text layer
                rotated_layer = txt_layer.rotate(angle, expand=False, resample=Image.BICUBIC)

                # --- COMPOSITING ---
                # Calculate coordinates to paste the center of rotated layer 
                # onto the center of the original image
                paste_x = (img_w - overlay_size) // 2
                paste_y = (img_h - overlay_size) // 2

                # Create a canvas the size of original image to hold the crop
                watermark_final = Image.new('RGBA', img.size, (0,0,0,0))
                
                # We paste the large rotated layer onto the canvas, utilizing the offsets
                # However, PIL paste doesn't handle negative offsets well with alpha compositing directly
                # So we crop the relevant part from rotated_layer or use alpha_composite directly with offsets
                
                # Simpler method: Crop the center of the rotated layer to match image size
                center_x = overlay_size // 2
                center_y = overlay_size // 2
                left = center_x - (img_w // 2)
                top = center_y - (img_h // 2)
                right = left + img_w
                bottom = top + img_h
                
                cropped_watermark = rotated_layer.crop((left, top, right, bottom))

                # Composite the watermark onto the image
                output_img = Image.alpha_composite(img, cropped_watermark)

                # Save
                save_path = os.path.join(OUTPUT_FOLDER, filename)
                output_img.save(save_path, "PNG")
                print(f"Processed: {filename}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print("Done!")

if __name__ == "__main__":
    add_watermark()