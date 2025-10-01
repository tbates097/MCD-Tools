import zipfile
import os
import xml.etree.ElementTree as ET
import json
import tkinter as tk
from tkinter import filedialog

def extract_mcd(mcd_path, extract_path):
    """Extracts the contents of an .MCD file to a specified directory."""
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(mcd_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        return zip_ref.namelist()

def parse_parameters(xml_path):
    """Parses the XML parameters file for all axes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    all_params = {}

    # Find all axes
    axes = root.findall(".//Axes/Axis")
    for axis in axes:
        axis_index = axis.get("Index")
        axis_params = {}
        
        # Get all parameters for this axis
        params = axis.findall(".//P")
        for param in params:
            name = param.get("n")
            value = param.text
            if name and value is not None:
                axis_params[name] = value
        
        all_params[f"axis_{axis_index}"] = axis_params

    # Get global parameters if they exist
    global_params = {}
    global_section = root.find(".//Global")
    if global_section is not None:
        params = global_section.findall(".//P")
        for param in params:
            name = param.get("n")
            value = param.text
            if name and value is not None:
                global_params[name] = value
    
    if global_params:
        all_params["global"] = global_params

    return all_params

def parse_xml_file(file_path):
    """Generic XML parser for any XML file in the MCD."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        def xml_to_dict(element):
            result = {}
            
            # Add attributes if they exist
            if element.attrib:
                result['@attributes'] = element.attrib
            
            # Add text content if it exists and isn't just whitespace
            if element.text and element.text.strip():
                result['#text'] = element.text.strip()
            
            # Process child elements
            for child in element:
                child_data = xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            
            return result
        
        return xml_to_dict(root)
    except ET.ParseError:
        return None

def process_mcd_file():
    """Main function to process an MCD file and generate JSON output."""
    # Select MCD file
    mcd_file = select_mcd_file()
    if not mcd_file:
        print("No file selected. Exiting...")
        return

    # Create temporary extraction directory
    extract_path = "extracted_mcd"
    output_json = "mcd_contents.json"

    try:
        # Extract MCD file
        print(f"Extracting {os.path.basename(mcd_file)}...")
        file_list = extract_mcd(mcd_file, extract_path)
        
        # Initialize dictionary to store all MCD contents
        mcd_contents = {
            "file_structure": file_list,
            "parameters": {},
            "xml_files": {},
            "other_files": {}
        }
        
        # Parse parameters if they exist
        params_path = os.path.join(extract_path, "config", "Parameters")
        if os.path.exists(params_path):
            print("Parsing parameters...")
            mcd_contents["parameters"] = parse_parameters(params_path)
        
        # Process all files in the MCD
        print("Processing all files...")
        for file_path in file_list:
            full_path = os.path.join(extract_path, file_path)
            if os.path.isfile(full_path):
                if file_path.lower().endswith('.xml'):
                    # Parse XML files
                    xml_content = parse_xml_file(full_path)
                    if xml_content:
                        mcd_contents["xml_files"][file_path] = xml_content
                else:
                    # For non-XML files, try to read as text if small enough
                    try:
                        if os.path.getsize(full_path) < 1024 * 1024:  # Only read files smaller than 1MB
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                mcd_contents["other_files"][file_path] = content
                        else:
                            mcd_contents["other_files"][file_path] = f"File size: {os.path.getsize(full_path)} bytes"
                    except (UnicodeDecodeError, IOError):
                        mcd_contents["other_files"][file_path] = "Binary file"
        
        # Save results to JSON
        print(f"Saving contents to {output_json}...")
        with open(output_json, "w", encoding="utf-8") as json_file:
            json.dump(mcd_contents, json_file, indent=4)
        
        print(f"Successfully saved MCD contents to {output_json}")
    
    except Exception as e:
        print(f"Error processing MCD file: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(extract_path):
            import shutil
            shutil.rmtree(extract_path, ignore_errors=True)

def select_mcd_file():
    """Opens a file dialog to select an MCD file."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    file_path = filedialog.askopenfilename(
        title="Select MCD File",
        filetypes=[("MCD files", "*.mcd"), ("All files", "*.*")]
    )
    
    return file_path if file_path else None

if __name__ == "__main__":
    process_mcd_file()