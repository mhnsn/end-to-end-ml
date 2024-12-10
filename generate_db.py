import os
import sqlite3
import random
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm  # Import tqdm for progress tracking
# from preprocess import parse_annotation_file

# Define the fields for the data dictionary (excluding the "speaker" field)
data_dict_fields = [
    "episode_id",   # Unique identifier for each episode
    "start_time",   # Start time of the transcript segment in the episode (in seconds)
    "end_time",     # End time of the transcript segment in the episode (in seconds)
    "text",          # The spoken content transcribed into text
    "themes"         # A field for themes related to the segment
]

def load_transcript(file_path):
    """Load the raw contents of the transcript file."""
    with open(file_path, 'r', encoding='utf-8', errors="replace") as f:
        return f.read()

def preprocess_text(text):
    """Preprocess the text by lowercasing and removing punctuation."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def basic_eda(text, stop_words=None, top_n=10):
    """Perform basic EDA on the text:
    - Count total words
    - Count unique words
    - Display top N frequent words (excluding stopwords if provided)
    """
    words = text.split()
    total_words = len(words)
    unique_words = len(set(words))

    # Filter out stopwords if provided
    if stop_words:
        words = [w for w in words if w not in stop_words]

    word_counts = Counter(words)
    most_common = word_counts.most_common(top_n)

    return {
        'total_words': total_words,
        'unique_words': unique_words,
        'top_words': most_common
    }

def named_entity_analysis(text):
    """Perform named entity recognition on the aggregated text and return a frequency counter of entities."""
    # Tokenize and POS tag
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    # Perform named entity chunking
    chunks = nltk.ne_chunk(pos_tags, binary=False)

    ne_counter = Counter()
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            ne_label = chunk.label()
            ne_text = " ".join(c[0] for c in chunk.leaves())
            ne_counter[ne_text] += 1
    return ne_counter

def create_database(db_name="podcast_data.db"):
    """Create a SQLite database to store the transcript data."""
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            episode_id TEXT,
            start_time REAL,
            end_time REAL,
            text TEXT,
            themes TEXT
        )
    ''')

    # Commit and close the connection
    conn.commit()
    conn.close()

def insert_data_into_db(data, db_name="podcast_data.db"):
    """Insert transcript data into the database."""
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Insert data into the table using tqdm for progress tracking
    for segment in tqdm(data, desc="Inserting data into database", unit="segment"):
        c.execute('''
            INSERT INTO transcripts (episode_id, start_time, end_time, text, themes)
            VALUES (?, ?, ?, ?, ?)
        ''', (segment['episode_id'], segment['start_time'], segment['end_time'], segment['text'], segment['themes']))

    # Commit and close the connection
    conn.commit()
    conn.close()

def parse_transcript_to_dict(file_path):
    """
    Parse the transcript into a dictionary of the required fields.
    
    Parameters:
    - file_path: Path to the transcript (.txt) file
    
    Returns:
    - A list of dictionaries with parsed fields.
    """
    raw_text = load_transcript(file_path)
    segments = []

    # Split the raw text into lines and process each segment
    lines = raw_text.strip().splitlines()

    # Process every three lines (start time, end time, and the corresponding text)
    for i in range(0, len(lines), 2):  # Now the time lines will be every two lines
        if i + 1 < len(lines):  # Ensure there is a corresponding text line
            # Parse start and end times from the first line (Start: X - End: Y)
            time_range = lines[i].strip().split(" - ")
            if len(time_range) == 2:
                start_time_str = time_range[0].replace("Start:", "").strip()  # Extract minutes:seconds.milliseconds
                end_time_str = time_range[1].replace("End:", "").strip()

                text = lines[i + 1].strip()  # The second line is the text

                try:
                    # Convert time strings to seconds (as float)
                    start_time = float(start_time_str.split(":")[0]) * 60 + float(start_time_str.split(":")[1])
                    end_time = float(end_time_str.split(":")[0]) * 60 + float(end_time_str.split(":")[1])

                    segment = {
                        "episode_id": os.path.basename(file_path),  # The filename as the episode ID
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text,
                        "themes": ""  # Placeholder for themes, can be updated later
                    }
                    segments.append(segment)
                except ValueError as e:
                    print(f"Error parsing times for episode: {file_path}, segment: {i}, error: {e}")
            # else:
            #     print(f"Skipping malformed line (missing 'Start - End' format): {lines[i]}")
    
    return segments

def build_data_dictionary_and_populate_db(transcripts_dir, db_name="podcast_data.db"):
    """Iterate over all transcript files and populate the database with data."""
    data = []
    
    # Get a list of all transcript files (assuming .txt files)
    all_files = [f for f in os.listdir(transcripts_dir) if f.endswith('.txt')]
    
    for file_name in tqdm(all_files, desc="Processing files", unit="file"):
        file_path = os.path.join(transcripts_dir, file_name)
        episode_data = parse_transcript_to_dict(file_path)
        data.extend(episode_data)
        
    # Insert the collected data into the database
    insert_data_into_db(data, db_name)

def main():
    # Directory containing transcripts
    transcripts_dir = "yt"  # Change this to directory path as needed

    # Create the database and table
    create_database()

    # Build the data dictionary and populate the database
    build_data_dictionary_and_populate_db(transcripts_dir)

    print("Data populated in the SQLite database successfully.")

if __name__ == "__main__":
    main()
