import os
import sys
import argparse

CONFIG_FILE = ".config"

def load_or_create_config():
    """
    Check if config.txt exists. If it does, read the folder path from it.
    If not, prompt the user for a folder path and save it to config.txt.
    """
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            folder_path = f.read().strip()
        if not folder_path:
            print("config.txt is empty. Please remove it and run again.")
            sys.exit(1)
        return folder_path
    else:
        folder_path = input("Enter the folder path containing .ner files: ").strip()
        if not folder_path or not os.path.isdir(folder_path):
            print("Invalid folder path.")
            sys.exit(1)
        # Save the folder path to config.txt
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(folder_path)
        return folder_path

def search_ner_files(folder_path, query):
    # Get all .ner files in the folder
    ner_files = [f for f in os.listdir(folder_path) if f.endswith(".ner")]

    if not ner_files:
        print(f"No .ner files found in {folder_path}.")
        return

    # Iterate over each .ner file and check for the query
    matching_files = []
    for ner_file in ner_files:
        file_path = os.path.join(folder_path, ner_file)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # Check if the query appears in this line (case-insensitive)
                if query.lower() in line.lower():
                    matching_files.append(ner_file)
                    break  # Found a match, no need to check further lines

    # Print matching files
    if matching_files:
        print("Files containing the query:")
        for mf in matching_files:
            print(mf)
    else:
        print("No files matched the query.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search NER files for a given query.")
    parser.add_argument("--query", "-q", help="The query string to search for in the .ner files.")

    args = parser.parse_args()
    query = args.query

    # If no query provided via command-line, prompt the user
    if not query:
        query = input("Enter the search query: ").strip()
        if not query:
            print("Query cannot be empty.")
            sys.exit(1)

    folder_name = load_or_create_config()
    search_ner_files(folder_name, query)
