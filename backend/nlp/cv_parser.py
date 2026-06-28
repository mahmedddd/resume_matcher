import fitz  # PyMuPDF

def parse_cv_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error parsing PDF: {e}")
    return text
