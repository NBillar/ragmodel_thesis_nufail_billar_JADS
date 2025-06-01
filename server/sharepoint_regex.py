from docx import Document
import re

def extract_sharepoint_link(text: str):
    """
    Extracts the first SharePoint link from the given text.
    """
    pattern = r"(https:\/\/[\w.-]*sharepoint\.com\/[^\s\"'<>]*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


