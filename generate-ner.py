import os
from transformers import pipeline
from tqdm import tqdm

# Initialize the NER pipeline with aggregation to combine subword tokens into full entities
ner_pipeline = pipeline("ner", aggregation_strategy="simple")

def parse_annotation_file(file_path):
    """
    Parse an annotation file to remove timestamps and empty lines.
    Return the cleaned text that can be passed to the NER model.

    Steps:
    - Reads the file line by line.
    - Skips lines starting with "Start:" since they contain timestamps.
    - Strips whitespace from each line.
    - Skips empty lines.
    - Joins the remaining lines into one large string.
    - Uses errors="replace" to handle decoding issues.
    """
    cleaned_lines = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Start:"):
                continue
            if not line:
                continue
            cleaned_lines.append(line)
    
    cleaned_text = " ".join(cleaned_lines)
    return cleaned_text

def process_folder_for_ner(folder_path):
    """
    Process all .txt annotation files in the given folder, perform NER, and save results as .ner files.

    Steps:
    - Lists all .txt files in the folder.
    - For each file, parse and clean the content.
    - If no content, skip.
    - Perform NER on the entire text at once with aggregation to get whole entities.
    - Save entities to a .ner file in a line-by-line format:
      ENTITY_TYPE,TEXT,START,END
    """
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    if not txt_files:
        print(f"No .txt files found in {folder_path}.")
        return
    
    print(f"Found {len(txt_files)} text files in {folder_path}.")

    for txt_file in tqdm(txt_files, desc="Processing files for NER"):
        file_path = os.path.join(folder_path, txt_file)
        ner_file_path = file_path.replace(".txt", ".ner")

        # Check if ner file already exists
        if os.path.exists(ner_file_path):
            print(f"Skipping {txt_file}, NER file already exists.")
            continue

        cleaned_content = parse_annotation_file(file_path)
        if not cleaned_content.strip():
            print(f"Skipping {txt_file} because it has no content after parsing.")
            continue
        
        # Perform NER on the entire content with aggregation
        entities = ner_pipeline(cleaned_content)

        # Save entities to .ner file
        # Format: ENTITY_TYPE,TEXT,START,END
        with open(ner_file_path, "w", encoding="utf-8") as nf:
            for ent in entities:
                # With aggregation_strategy="simple", 'entity_group' provides the entity label
                ent_type = ent.get("entity_group", "UNKNOWN")
                start_idx = ent["start"]
                end_idx = ent["end"]
                # The text can be extracted directly
                ent_text = cleaned_content[start_idx:end_idx]
                nf.write(f"{ent_type},{ent_text},{start_idx},{end_idx}\n")

if __name__ == "__main__":
    folder_name = input("Enter the folder path containing the .txt transcripts: ").strip()
    if not os.path.isdir(folder_name):
        print("Invalid folder path.")
    else:
        process_folder_for_ner(folder_name)
