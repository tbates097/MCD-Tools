import zipfile
import os
import shutil
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
from datetime import datetime

def main():
    # Ask user to select MCD file
    root = tk.Tk()
    root.withdraw()
    
    mcd_path = filedialog.askopenfilename(
        title="Select MCD file to examine",
        filetypes=[("MCD files", "*.mcd"), ("All files", "*.*")]
    )
    
    if not mcd_path:
        print("No file selected")
        return

    # Create output filename
    base_name = os.path.splitext(os.path.basename(mcd_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{base_name}_config_contents_{timestamp}.txt"
    
    # Create temp directory
    temp_dir = "temp_mcd_extract"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    try:
        print(f"Extracting {os.path.basename(mcd_path)}...")
        
        # Extract MCD
        with zipfile.ZipFile(mcd_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Open output file
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(f"MCD Config Files Examination\n")
            out_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            out_file.write(f"Source MCD: {mcd_path}\n")
            out_file.write("="*80 + "\n\n")
            
            # Process config directory
            config_dir = os.path.join(temp_dir, "config")
            if os.path.exists(config_dir):
                for file_name in sorted(os.listdir(config_dir)):
                    file_path = os.path.join(config_dir, file_name)
                    
                    # Skip SignalLog.dat
                    if "SignalLog.dat" in file_path:
                        continue
                        
                    out_file.write(f"\nFile: {file_name}\n")
                    out_file.write("-"*80 + "\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8-sig') as f:
                            content = f.read()
                            out_file.write(content)
                            out_file.write("\n")
                    except UnicodeDecodeError:
                        out_file.write("[Binary file - contents not shown]\n")
                    
                    out_file.write("\n" + "="*80 + "\n")
        
        print(f"Examination complete! Results saved to: {output_path}")
        
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()