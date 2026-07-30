import pdfplumber
import sys

pdf_path = r"D:/Repositories/Machine_Unlearning/Unlearn-Simple/docs/Simplicity Prevails-Rethinking Negative Preference.pdf"
output_path = r"D:/Repositories/Machine_Unlearning/Unlearn-Simple/docs/paper_text.txt"

with pdfplumber.open(pdf_path) as pdf:
    with open(output_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                f.write(f"=== PAGE {i+1} ===\n")
                f.write(text)
                f.write("\n\n")
print(f"Extracted {len(pdf.pages)} pages to {output_path}")
