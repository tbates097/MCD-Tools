import xml.dom.minidom
import codecs

def convert_machine_setup_to_xml(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"First 100 characters of content: {repr(content[:100])}")
    
    try:
        # Find the XML content between the quotes after "config/MachineSetupData": 
        start_marker = '"config/MachineSetupData": "'
        end_marker = '",'  # Note: added comma to get the exact end
        
        start_idx = content.find(start_marker)
        if start_idx == -1:
            print(f"Could not find start marker '{start_marker}' in the file")
            return
            
        start_idx += len(start_marker)
        end_idx = content.find(end_marker, start_idx)
        
        if end_idx == -1:
            end_idx = content.rfind('"')  # Fallback to last quote if no comma found
        
        if end_idx <= start_idx:
            print(f"Invalid end marker position. Start: {start_idx}, End: {end_idx}")
            return
            
        # Extract the XML string
        xml_string = content[start_idx:end_idx]
        print(f"Found XML content of length: {len(xml_string)}")
        print(f"First 100 characters of XML: {repr(xml_string[:100])}")
        
        # Decode the string literals (handles \n, \", \ufeff, etc.)
        xml_string = codecs.decode(xml_string, 'unicode_escape')
        
        # Remove the BOM if present
        if xml_string.startswith('\ufeff'):
            xml_string = xml_string[1:]
        
        # Parse and pretty print the XML
        dom = xml.dom.minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent='  ')
        
        # Write the formatted XML to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
            
        print(f"Successfully converted {input_file} to {output_file}")
        
    except xml.parsers.expat.ExpatError as e:
        print(f"Error parsing XML: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    input_file = "Machine Setup-PRO165SL.txt"
    output_file = "machine_setup-PRO165SL.xml"
    convert_machine_setup_to_xml(input_file, output_file) 