"""Create a simple PDF for testing."""

import fitz  # PyMuPDF

def create_test_pdf(filename, text):
    """Create a PDF with the given text."""
    doc = fitz.open()
    page = doc.new_page()
    # Insert text at the top-left
    page.insert_text((50, 50), text, fontsize=12)
    doc.save(filename)
    doc.close()
    print(f"Created {filename}")

if __name__ == "__main__":
    create_test_pdf("Data/test1.pdf", "This is a test document about apples. Apples are fruits.")
    create_test_pdf("Data/test2.pdf", "This document discusses oranges. Oranges are citrus fruits.")