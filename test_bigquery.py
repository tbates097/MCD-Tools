# test_gtse_translator.py
try:
    print("Testing GTSEQueryTranslator creation...")
    from GTSE_query import GTSEQueryTranslator  # Note: filename with space might need quotes
    
    translator = GTSEQueryTranslator()
    print("✓ GTSEQueryTranslator created successfully")
    
except Exception as e:
    print(f"✗ Error creating GTSEQueryTranslator: {e}")
    import traceback
    traceback.print_exc()