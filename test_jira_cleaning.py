import re
import pandas as pd

def clean_jira_text(text: str) -> str:
    """
    Clean Jira formatting markup from text
    
    Args:
        text: Raw text with Jira markup
        
    Returns:
        Cleaned text
    """
    if not text or pd.isna(text):
        return text
    
    clean_text = str(text)
    
    # Remove Jira color formatting tags like {color:#000000} or {color:red}
    clean_text = re.sub(r'\{color[^}]*\}', '', clean_text)
    
    # Remove other Jira formatting tags like {panel}, {code}, {quote}, etc.
    clean_text = re.sub(r'\{[^}]*\}', '', clean_text)
    
    # Replace various forms of line endings with spaces
    clean_text = clean_text.replace('\\r\\n', ' ')  # Escaped version
    clean_text = clean_text.replace('\r\n', ' ')    # Actual line breaks
    clean_text = clean_text.replace('\\n', ' ')     # Just newlines
    clean_text = clean_text.replace('\n', ' ')      # Actual newlines
    clean_text = clean_text.replace('\\r', ' ')     # Just carriage returns
    clean_text = clean_text.replace('\r', ' ')      # Actual carriage returns
    
    # Remove table formatting pipes and other table markup
    clean_text = re.sub(r'\s*\|\s*', ' ', clean_text)  # Table cell separators
    clean_text = re.sub(r'\|\|', ' ', clean_text)      # Table headers
    
    # Remove Jira links and markup
    clean_text = re.sub(r'\[([^\]]*)\|([^\]]*)\]', r'\1', clean_text)  # [text|link] -> text
    clean_text = re.sub(r'\[([^\]]*)\]', r'\1', clean_text)            # [text] -> text
    
    # Remove common Jira markup patterns
    clean_text = re.sub(r'\*([^*]*)\*', r'\1', clean_text)  # *bold* -> bold
    clean_text = re.sub(r'_([^_]*)_', r'\1', clean_text)    # _italic_ -> italic
    clean_text = re.sub(r'\+([^+]*)\+', r'\1', clean_text)  # +underline+ -> underline
    clean_text = re.sub(r'-([^-]*)-', r'\1', clean_text)    # -strikethrough- -> strikethrough
    clean_text = re.sub(r'\^([^^]*)\^', r'\1', clean_text)  # ^superscript^ -> superscript
    clean_text = re.sub(r'~([^~]*)~', r'\1', clean_text)    # ~subscript~ -> subscript
    
    # Remove code blocks and inline code
    clean_text = re.sub(r'\{\{([^}]*)\}\}', r'\1', clean_text)  # {{code}} -> code
    clean_text = re.sub(r'`([^`]*)`', r'\1', clean_text)        # `code` -> code
    
    # Remove headers (h1., h2., etc.)
    clean_text = re.sub(r'^h[1-6]\.\s*', '', clean_text, flags=re.MULTILINE)
    
    # Remove quote blocks (bq.)
    clean_text = re.sub(r'^bq\.\s*', '', clean_text, flags=re.MULTILINE)
    
    # Remove list markers
    clean_text = re.sub(r'^\*+\s*', '', clean_text, flags=re.MULTILINE)  # Bullet lists
    clean_text = re.sub(r'^#+\s*', '', clean_text, flags=re.MULTILINE)   # Numbered lists
    clean_text = re.sub(r'^-+\s*', '', clean_text, flags=re.MULTILINE)   # Dash lists
    
    # Clean up multiple spaces and normalize whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # Trim whitespace
    clean_text = clean_text.strip()
    
    return clean_text

# Test with the user's example
original_text = "Description: Hello Clive,\\r\\n\\r\\nWe are planning to install an XR3 instead of a NPAQ on the S/N 17304 system (the 10th and 11th of april).\\r\\n\\r\\nAfter that, we would like to perform a remote tuning session on this setup.\\r\\n\\r\\nCould you please tell me if you would be OK to do this on the Monday the 15th of apr..."

print("=== Jira Text Cleaning Test ===")
print("\nOriginal text:")
print(repr(original_text))
print("\nOriginal text (readable):")
print(original_text)

cleaned_text = clean_jira_text(original_text)

print("\nCleaned text:")
print(repr(cleaned_text))
print("\nCleaned text (readable):")
print(cleaned_text)

print("\n=== Additional Test Cases ===")

test_cases = [
    "{color:#ff0000}Error message{color}",
    "*Bold text* and _italic text_",
    "[Link text|http://example.com]",
    "Line 1\\r\\nLine 2\\r\\nLine 3",
    "Cell 1 | Cell 2 | Cell 3",
    "h1. Header Text",
    "* Bullet point 1\\r\\n* Bullet point 2",
    "{{code block}}",
    "Regular text with {panel} markup {panel}"
]

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}:")
    print(f"Original: {repr(test)}")
    print(f"Cleaned:  {repr(clean_jira_text(test))}") 