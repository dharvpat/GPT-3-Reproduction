import bz2
import re
import os
from nltk.corpus import words
import nltk

# Download the NLTK words corpus if not already present
nltk.download('words')

valid_words = set(words.words())  # A set of valid English words

def is_nonsensical_sentence(sentence):
    # Heuristic 1: Sentence length
    words_in_sentence = sentence.split()
    if len(words_in_sentence) < 3 or len(words_in_sentence) > 50:
        return True  # Too short or too long sentences are likely nonsensical

    # Heuristic 2: High ratio of non-dictionary words
    valid_word_count = sum(1 for word in words_in_sentence if word in valid_words)
    if valid_word_count / len(words_in_sentence) < 0.5:  # Less than 50% valid words
        return True

    # Heuristic 3: Detect nonsensical repetitive patterns (e.g., "asdfasdfasdf")
    if re.search(r'(\b\w{4,}\b)\1{2,}', sentence):  # Repeated word patterns
        return True

    # If none of the heuristics match, the sentence is considered valid
    return False

def enhanced_clean_text_v2(text):
    # Step 1: Remove XML-like and HTML-like tags (e.g., <namespace>, </namespace>)
    text = re.sub(r'<[^>]+>', '', text)

    # Step 2: Remove citation-related text blocks (e.g., 'cite book', 'cite journal', etc.)
    citation_patterns = [
        r'cite journal.*?(?=\n|$)',  # Remove 'cite journal' lines until the end of the line or paragraph
        r'cite book.*?(?=\n|$)',     # Remove 'cite book' lines until the end of the line or paragraph
        r'cite encyclopedia.*?(?=\n|$)',  # Remove 'cite encyclopedia'
        r'cite magazine.*?(?=\n|$)', # Remove 'cite magazine'
    ]
    for pattern in citation_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Step 3: Remove HTML-like formatting attributes (e.g., 'style="..."', 'scope="row"', etc.)
    # Remove attributes like style="..." and scope="..."
    text = re.sub(r'stylequot[^"]+quot', '', text)  # Remove 'stylequot...quot' patterns
    text = re.sub(r'scopequot[^"]+quot', '', text)  # Remove 'scopequot...quot' patterns

    # Step 4: Remove "r category shell" and similar wiki-related artifacts
    text = re.sub(r'\br(?:\s+category|shell|from)\b.*?(?=\n|$)', '', text, flags=re.IGNORECASE)  # Remove lines starting with 'r' and related wiki formatting

    # Step 5: Remove standalone letters and meaningless words
    text = re.sub(r'\br\b', '', text)  # Remove standalone 'r'
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove single letters (like 'r')

    # Step 6: Remove numeric metadata and key-value metadata
    text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
    text = re.sub(r'\b(?:[A-Za-z0-9]+:)[^\s]+\b', '', text)  # Remove things like `key:value`

    # Step 7: Remove special characters and excess spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove anything that's not a letter or space
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space

    # Step 8: Lowercase all text
    text = text.lower()

    return text

def clean_and_filter_sentences(text):
    sentences = text.split("\n")  # Split text into sentences by newlines
    filtered_sentences = []

    for sentence in sentences:
        # Remove lines starting with 'redirect'
        if sentence.lower().startswith('redirect'):
            continue

        cleaned_sentence = enhanced_clean_text_v2(sentence)  # Apply your existing cleaning function
        if cleaned_sentence and not is_nonsensical_sentence(cleaned_sentence):  # Filter nonsensical sentences
            filtered_sentences.append(cleaned_sentence)

    return " ".join(filtered_sentences)  # Join valid sentences back together

# Optimized function to extract and preprocess the Wikipedia dataset
def wikipedia_preprocessing(raw_data_path, processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    processed_data_path = os.path.join(processed_dir, "wikipedia_processed.txt")

    print(f"Processing and cleaning Wikipedia dataset from {raw_data_path}...")
    
    with bz2.open(raw_data_path, 'rt') as raw_file, open(processed_data_path, 'w') as processed_file:
        # Process line by line to minimize memory usage
        for line in raw_file:
            # Clean the line
            cleaned_line = clean_and_filter_sentences(line)
            
            # Write cleaned line to processed file
            if cleaned_line:  # Only write non-empty lines
                processed_file.write(cleaned_line + "\n")
    
    print(f"Processed data saved to {processed_data_path}.")
    return processed_data_path