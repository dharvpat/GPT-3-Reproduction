# Data Preparation Guide

This guide explains how to acquire, preprocess, and manage the datasets required for training the GPT-3 models.

---

## Overview

The models are trained on a large corpus of text data, including:

- Common Crawl
- WebText2
- Books Corpus
- Wikipedia

---

## 1. Data Acquisition

### 1.1. Common Crawl

- **Description**: A dataset containing petabytes of web page data.
- **Download Script**:

  ```bash
  python3 scripts/download_datasets.py --dataset common_crawl
  ```

### 1.2. WebText2

- **Description**: An OpenAI-curated dataset based on web content.
- **Download**: If unavailable, consider using OpenWebText as an alternative.

### 1.3. Book Corpus

- **Description**: A collection of thousands of books from various genres.
- **Download**: Obtain from authorized sources or use publicly available datasets like BookCorpus2.

### 1.4. Wikipedia

- **Description**: The entire English WIkipedia Collection. Massive dataset with very good data quality.
- **Download**:
```bash
python3 scripts/download_datasets.py --dataset wikipedia
```

## 2. Data Preprocessing

### 2.1. Text Cleaning

- **Remove HTML Tags**: Clean web data to extract plain text.
- **Normalize Text**: Convert to lowercase, remove special characters.
- **Remove Duplicates**: Identify and eliminate duplicate content.

### 2.2. Tokenization

- **Tokenizer**: Use Byte Pair Encoding (BPE) with a vocabulary size of 50,000 tokens.
- **Scripts**:
```bash
python3 scripts/preprocess_data.py --tokenze
```
- **Output**: Tokenized, processed dataset stored in `data/processed`

## 3. Data Organization

- **Raw Data**: Raw Data is stored in `data/raw/`
- **Processed Data**: Processed data is stored in `data/processed/`
- **Tokenizer**: Files that are used in tokenizers and stored in `data/tokenizers/`

## 4. Handling Large Datasets

### 4.1. Storage Considereations

- Ensure you have sufficient HDD storage
- We will use binary files to minimize overhead

### 4.2. Streaming Data
- For very large datasets like the wikipedia library, we implement data streaming to minimize RAM requirements.

## 5. Scripts for Automation

- Download Automation:
```bash
python3 scripts/download_datasets.py --all
```

- Preprocess Automation:
```bash
python3 scripts/preprocess_data.py --all
```