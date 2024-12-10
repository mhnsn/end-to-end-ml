import os

def parse_annotation_file(file_path):
    """
    Parse an annotation file to remove timestamps and empty lines.
    Return the cleaned text that can be passed to a tokenizer or summarization model.
    
    Steps:
    - Reads the file line by line.
    - Skips lines that start with "Start:" since they contain timestamps.
    - Strips leading and trailing whitespace from each line.
    - Skips empty lines after stripping.
    - Joins all kept lines into one large string for processing.
    - Uses errors="replace" to handle any invalid UTF-8 characters gracefully.
    """
    cleaned_lines = []
    # Open file with UTF-8 encoding and replace invalid characters to avoid UnicodeDecodeError
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            # Skip timestamp lines
            if line.startswith("Start:"):
                continue
            # Skip empty lines
            if not line:
                continue
            # Keep the cleaned line
            cleaned_lines.append(line)
    
    # Join all lines into one continuous string
    cleaned_text = " ".join(cleaned_lines)
    return cleaned_text