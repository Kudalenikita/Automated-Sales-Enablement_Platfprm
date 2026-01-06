# automated_sales_enablement/utils/utils.py

def normalize_text(text):
    return text.lower().strip()

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]