import pypdf
import os

pdf_path = '/root/Commune-Staff-Chatbot/sample_data/cam_nang_chinh_quyen.pdf'
txt_path = '/root/Commune-Staff-Chatbot/sample_data/cam_nang_chinh_quyen.txt'

try:
    reader = pypdf.PdfReader(pdf_path)
    full_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text.append(text)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_text))
    
    print(f"✅ Extracted text to {txt_path}")
except Exception as e:
    print(f"❌ Error: {e}")
