import os
from transformers import pipeline
from tqdm import tqdm

# Load the summarization pipeline. 
# Using "facebook/bart-large-cnn" as the model and device=0 to use the first available GPU if present.
# If torch.cuda.is_available() is False, this will run on CPU even with device=0.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

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

def determine_max_length(text, default_max=500, min_length=50):
    """
    Determine the max_length for summarization based on the text length.
    
    Logic:
    - We approximate the token count by splitting on whitespace (words).
    - If the word count is less than default_max, we use the word count as max_length 
      to avoid warnings about max_length > input_length.
    - Ensure max_length is never less than min_length to avoid potential errors where min_length > max_length.
    
    This helps reduce warnings from the model about chosen max_length being larger than the input.
    """
    word_count = len(text.split())
    # max_len is at most default_max, but at least min_length, and also not larger than word_count if it's smaller
    max_len = min(default_max, max(word_count, min_length))
    return max_len

def summarize_in_chunks(text, chunk_size=1024, default_max_length=500, min_length=50):
    """
    Summarize text by splitting it into chunks if it's too long.
    Returns a single summary string.
    
    Steps:
    - If the text length is within chunk_size, summarize directly.
    - Otherwise, split into chunks of size chunk_size (in characters).
    - Summarize each chunk individually, then combine those summaries.
    - If the combined summary is still large, call this function recursively to summarize the summary.
    - Use determine_max_length to pick an appropriate max_length parameter dynamically for each summarization call.
    """
    text = text.strip()
    if not text:
        return "No content to summarize."

    # Dynamically determine max_length for the entire text
    max_length = determine_max_length(text, default_max=default_max_length, min_length=min_length)

    if len(text) <= chunk_size:
        # Text is short enough to summarize in one go
        return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    else:
        # Text is long, so we split into chunks
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        chunk_summaries = []
        
        for chunk in text_chunks:
            chunk = chunk.strip()
            if chunk:
                # Determine max_length for this chunk
                chunk_max_length = determine_max_length(chunk, default_max=default_max_length, min_length=min_length)
                # Summarize the chunk
                cs = summarizer(chunk, max_length=chunk_max_length, min_length=min_length, do_sample=False)[0]['summary_text']
                chunk_summaries.append(cs)
        
        # Combine the chunk summaries
        combined_text = " ".join(chunk_summaries)
        
        # If the combined summary is still large, summarize it again
        if len(combined_text) > chunk_size:
            return summarize_in_chunks(combined_text, chunk_size, default_max_length, min_length)
        else:
            # Final pass on the combined summary
            final_max_length = determine_max_length(combined_text, default_max=default_max_length, min_length=min_length)
            return summarizer(combined_text, max_length=final_max_length, min_length=min_length, do_sample=False)[0]['summary_text']

def summarize_text(text, chunk_size=1024, intermediate_summary_length=500, final_summary_length=500):
    """
    Summarizes the input text by:
    1. Splitting the text into intermediate chunks (character-based) and summarizing each chunk.
    2. Combining those intermediate summaries into one text.
    3. Summarizing the combined text again (final summary).
    4. If the final input is still too long, it will be handled by summarize_in_chunks recursively.
    
    Steps:
    - Strip the text.
    - Split the original text into chunks of size chunk_size.
    - Summarize each chunk with a dynamically chosen max_length (up to intermediate_summary_length).
    - Combine the intermediate summaries and perform a final summarization pass using summarize_in_chunks with 
      final_summary_length as the default_max_length.
    
    By increasing from 200 to 500 as default_max_length, we allow longer summaries. The code still adjusts 
    max_length if the input is shorter, to reduce warning messages and produce more sensible output.
    """
    text = text.strip()
    if not text:
        return "No content to summarize."
    
    # Split into intermediate chunks
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    chunk_summaries = []
    
    # Summarize each intermediate chunk
    for chunk in text_chunks:
        chunk = chunk.strip()
        if chunk:
            # Determine max_length for intermediate chunks
            chunk_max_length = determine_max_length(chunk, default_max=intermediate_summary_length, min_length=50)
            cs = summarizer(chunk, max_length=chunk_max_length, min_length=50, do_sample=False)[0]['summary_text']
            chunk_summaries.append(cs)
    
    if not chunk_summaries:
        return "No content to summarize."
    
    # Combine all intermediate summaries
    combined_summary_input = " ".join(chunk_summaries)
    
    # Perform the final summarization pass
    # This will also chunk again if needed
    final_summary = summarize_in_chunks(
        combined_summary_input, 
        chunk_size=chunk_size, 
        default_max_length=final_summary_length, 
        min_length=50
    )
    return final_summary

def summarize_folder(folder_path):
    """
    Summarize all .txt annotation files in the given folder and save the results as .summary files.
    
    Steps:
    - List all .txt files in the folder.
    - For each .txt file, derive a corresponding .summary file path.
    - If the .summary file already exists, skip processing to avoid overwriting.
    - Parse and clean the annotation file.
    - If there's no content after parsing, skip it.
    - Otherwise, summarize the content and write the result to the .summary file.
    """
    # Get all .txt files from the given folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    if not txt_files:
        print(f"No .txt files found in {folder_path}.")
        return
    
    print(f"Found {len(txt_files)} text files in {folder_path}.")

    # Process each .txt file
    for txt_file in tqdm(txt_files, desc="Summarizing files"):
        file_path = os.path.join(folder_path, txt_file)
        summary_file_path = file_path.replace(".txt", ".summary")

        # Check if summary file already exists to save time
        if os.path.exists(summary_file_path):
            print(f"Skipping {txt_file}, summary already exists.")
            continue

        # Parse and clean the annotation file
        cleaned_content = parse_annotation_file(file_path)

        # If there's nothing to summarize after parsing, skip it
        if not cleaned_content.strip():
            print(f"Skipping {txt_file} because it has no content after parsing.")
            continue
        
        # Summarize the cleaned content
        summary = summarize_text(
            cleaned_content, 
            chunk_size=1024, 
            intermediate_summary_length=500,  # Increased from 200 to 500
            final_summary_length=500           # Increased from 200 to 500
        )
        
        # Save the final summary to a .summary file
        with open(summary_file_path, "w", encoding="utf-8") as sf:
            sf.write(summary)

if __name__ == "__main__":
    # Prompt user for the folder path containing .txt transcripts
    folder_name = input("Enter the folder path containing the .txt transcripts: ").strip()
    if not os.path.isdir(folder_name):
        print("Invalid folder path.")
    else:
        summarize_folder(folder_name)
