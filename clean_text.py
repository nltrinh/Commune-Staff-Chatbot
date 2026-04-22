import re
import os

input_path = '/root/Commune-Staff-Chatbot/sample_data/cam_nang_chinh_quyen.txt'
output_path = '/root/Commune-Staff-Chatbot/sample_data/cam_nang_chinh_quyen_clean.txt'

def clean_text(text):
    # Keep alphanumeric, common punctuation, and Vietnamese characters
    # Vietnamese ranges: \u00C0-\u1EF9
    # Alphanumeric and common symbols: a-zA-Z0-9\s.,!?:;()\"\'\-
    # We'll use a more permissive approach: remove control characters and very weird symbols
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', text)
    # Also replace potential NaN causing patterns if any (though unlikely in text)
    return text

try:
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cleaned = clean_text(content)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"✅ Cleaned text saved to {output_path}")
except Exception as e:
    print(f"❌ Error: {e}")
