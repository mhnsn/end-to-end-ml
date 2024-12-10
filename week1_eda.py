import os
import random
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from preprocess import parse_annotation_file

def load_transcript(file_path):
    return parse_annotation_file(file_path)

def preprocess_text(text):
    """Preprocess the text by lowercasing and removing punctuation.
    Note that lowercasing will mangle named entity recognition, so undo it to get accurate initial reads.    
    """

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

def main():
    # Directory containing your transcripts
    transcripts_dir = "yt"

    # Get a list of all transcript files (assuming .txt files)
    all_files = [f for f in os.listdir(transcripts_dir) if f.endswith('.txt')]
    
    # Set a fixed random seed for reproducibility
    random.seed(37)

    # Randomly select 10 files using the fixed seed
    if len(all_files) < 10:
        raise ValueError("Not enough episodes (.txt files) in the directory to select 10.")
    
    selected_files = random.sample(all_files, 10)
    
    print("Selected Episodes:")
    for sf in selected_files:
        print(sf)
    print("\nPerforming EDA on these episodes...\n")

    # Base stopwords from NLTK
    base_stopwords = set(stopwords.words('english'))
    # Additional custom stopwords
    extra_stopwords = {"thats", "well", "theres", "us", "im", "like", "chapter", "verse", "would"}
    # Merge them into one set
    stop_words = base_stopwords.union(extra_stopwords)

    # Aggregate stats across the selected episodes
    aggregate_stats = {
        'total_words_all': 0,
        'total_unique_words': set(),
        'word_counter': Counter(),
        'all_processed_text': []
    }

    for file_name in selected_files:
        file_path = os.path.join(transcripts_dir, file_name)
        raw_text = load_transcript(file_path)
        processed_text = preprocess_text(raw_text)

        eda_results = basic_eda(processed_text, stop_words=stop_words, top_n=10)

        # Print stats for this episode
        print(f"--- EDA for {file_name} ---")
        print(f"Total words: {eda_results['total_words']}")
        print(f"Unique words: {eda_results['unique_words']}")
        print("Top 10 words (excluding stopwords):")
        for word, count in eda_results['top_words']:
            print(f"{word}: {count}")
        print("\n")

        # Update aggregate stats
        words = processed_text.split()
        aggregate_stats['total_words_all'] += len(words)
        aggregate_stats['total_unique_words'].update(words)
        # Filter words again for counting without stopwords
        filtered_words = [w for w in words if w not in stop_words]
        aggregate_stats['word_counter'].update(filtered_words)
        aggregate_stats['all_processed_text'].extend(words)

    # Print aggregate statistics
    print("=== Aggregate Statistics (Selected 10 Episodes) ===")
    total_episodes = 10
    total_words_all = aggregate_stats['total_words_all']
    total_unique_count = len(aggregate_stats['total_unique_words'])

    # Average words per episode
    avg_words_per_episode = total_words_all / total_episodes

    # Vocabulary diversity (unique_words / total_words)
    vocab_diversity = total_unique_count / total_words_all if total_words_all > 0 else 0

    # Frequency of filler words
    filler_words = {"um", "uh", "like"}
    filler_count = sum(w in filler_words for w in aggregate_stats['all_processed_text'])

    # Named entity analysis on the aggregated text
    aggregated_text = " ".join(aggregate_stats['all_processed_text'])
    ne_counter = named_entity_analysis(aggregated_text)
    # Top 10 named entities
    top_named_entities = ne_counter.most_common(10)

    # Topic frequency (using top aggregated words)
    top_aggregated = aggregate_stats['word_counter'].most_common(10)

    print(f"Total words across selected episodes: {total_words_all}")
    print(f"Total unique words across selected episodes: {total_unique_count}")
    print(f"Average words per episode: {avg_words_per_episode:.2f}")
    print(f"Vocabulary diversity (unique/total): {vocab_diversity:.4f}")

    print("\nFiller Words Frequency:")
    for fw in filler_words:
        fw_count = aggregate_stats['all_processed_text'].count(fw)
        print(f"{fw}: {fw_count}")

    print("\nTop 10 Named Entities:")
    for ent, cnt in top_named_entities:
        print(f"{ent}: {cnt}")

    print("\nTop 10 'Topics' (based on top words excluding stopwords):")
    for word, count in top_aggregated:
        print(f"{word}: {count}")

if __name__ == "__main__":
    # Ensure NLTK resources are available; if not, uncomment:
    # nltk.download('stopwords')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('maxent_ne_chunker')
    # nltk.download('words')
    # nltk.download('punkt')
	# nltk.download('averaged_perceptron_tagger_eng')
    # nltk.download('maxent_ne_chunker_tab')
    # nltk.download('punkt_tab')

    main()