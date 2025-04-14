from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import requests
import pandas as pd
from itertools import combinations
from io import BytesIO
import re
from datetime import datetime
import time
import hashlib
import os
import shutil # To clear the cache manually

# version 3:Here are several optimizations to improve the execution time:
#Key optimizations:
#1. Parallel URL fetching using ThreadPoolExecutor
#2. Model warm-up to avoid first-run overhead
#3. Batch processing of sentence embeddings
#4. Increased batch size for encoding
#5. Add caching for URL content:

start_time = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
# shutil.rmtree("url_cache")  # Deletes entire cache directory
def extract_text(url):
    # Create cache directory if it doesn't exist
    cache_dir = "url_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create cache filename from URL
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{url_hash}.txt")
    
    # Try to read from cache first
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # If not in cache, fetch and process URL
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator=' ')
        processed_text = re.sub(r'\s+', ' ', text).strip()
        
        # Save to cache
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(processed_text)
            
        return processed_text
    except Exception as e:
        return ""

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if 20 < len(s.strip()) < 500]  # Add upper limit

def generate_similarity_excel(urls, top_n=4):
    process_times = {}  # Track timing for each process
    # Cache model encoding
    t_start = time.time()
    model.encode(['test'], convert_to_tensor=True)  # Warm up GPU/CPU
    process_times['model_warmup'] = time.time() - t_start

    # Parallel URL fetching
    t_start = time.time()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        texts = list(executor.map(extract_text, urls))
    process_times['url_fetching'] = time.time() - t_start

    # Batch process sentences #Process each URL and split into sentences
    t_start = time.time()
    page_contents = {}
    page_sentences = {}
    page_labels = {}
    sentence_id_locations = {}  # Track locations for hyperlinking
    for idx, text in enumerate(texts):
        label = f"Page{chr(65 + idx)}"
        sentences = split_into_sentences(text)
        page_contents[label] = text
        page_sentences[label] = sentences
        page_labels[urls[idx]] = label
    process_times['sentence_processing'] = time.time() - t_start

    # Pre-compute all embeddings at once
    t_start = time.time()
    all_sentences = []
    sentence_map = {}
    sentence_indices = {}  # Track indices of sentences for each label
    current_idx = 0
    for label, sentences in page_sentences.items():
        sentence_indices[label] = (current_idx, current_idx + len(sentences))
        current_idx += len(sentences)
        for sent in sentences:
            all_sentences.append(sent)
            sentence_map[sent] = label
    
    all_embeddings = model.encode(all_sentences, convert_to_tensor=True, batch_size=32)
    process_times['embeddings_generation'] = time.time() - t_start
    
    # Generate Excel generation report
    t_start = time.time()
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    workbook = writer.book
    # Create a format for hyperlinks
    link_format = workbook.add_format({
        'font_color': 'blue',
        'underline': True,
        'bold': True
    })

    labels = list(page_sentences.keys())
    # Create a dictionary to store all sheet data
    all_sheet_data = {}
    # Process sequential pairs (A->B, B->C, C->D, D->A)
    sequential_pairs = [(labels[i], labels[(i + 1) % len(labels)]) for i in range(len(labels))]
    # Process combination pairs (A->C, B->D), while avoiding duplicates (including reverse pairs)
    combination_pairs = []
    for label1, label2 in combinations(labels, 2):
        # Skip if pair or its reverse exists in sequential pairs
        if (label1, label2) not in sequential_pairs and \
           (label2, label1) not in sequential_pairs and \
           (label1, label2) not in combination_pairs and \
           (label2, label1) not in combination_pairs:
            combination_pairs.append((label1, label2))
    # Combine both types of pairs
    all_pairs = sequential_pairs + combination_pairs
    for label1, label2 in all_pairs:
        #label1 = labels[i]
        #label2 = labels[(i + 1) % len(labels)]

        # Get pre-computed embeddings for both pages
        start_idx1, end_idx1 = sentence_indices[label1]
        start_idx2, end_idx2 = sentence_indices[label2]
        embeddings1 = all_embeddings[start_idx1:end_idx1]
        embeddings2 = all_embeddings[start_idx2:end_idx2]
        sentences1 = page_sentences[label1]
        sentences2 = page_sentences[label2]

        sheet_data = []
        pageId1 = label1.replace("Page", "")
        pageId2 = label2.replace("Page", "")
        sheet_name = f"{pageId1}vs{pageId2}"[:31]

        for idx1, (sentence1, emb1) in enumerate(zip(sentences1, embeddings1)):
            sentId1 = f"{pageId1}S{idx1 + 1}"
            scores = util.pytorch_cos_sim(emb1.unsqueeze(0), embeddings2)[0]
            top_indices = scores.argsort(descending=True)[:top_n]

            data_row = {
                f"{label1}-SentenceID": sentId1,
                f"{label1}-SentenceText": sentence1,
            }
            for j, idx2 in enumerate(top_indices):
                if idx2 is None:
                    continue
                try:
                    sentId2 = f"{pageId2}S{idx2.item() + 1}"
                    sentence2 = sentences2[idx2]
                    score1v2 = scores[idx2].item()

                    data_row[f"Matched{pageId1}vs{pageId2} Top{j + 1}-Score"] = f"{score1v2:.2f}"
                    data_row[f"Matched{pageId1}vs{pageId2} Top{j + 1}-SentenceID"] = sentId2
                    data_row[f"Matched{pageId1}vs{pageId2} Top{j + 1}-Text"] = sentence2

                except (IndexError, AttributeError) as e:
                    continue

            sheet_data.append(data_row)

        df = pd.DataFrame(sheet_data)
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]
        # Store sheet_data for later use
        all_sheet_data[sheet_name] = sheet_data

        # Track row locations for each sentence ID (for hyperlinking later)
        for row_idx, sent_row in enumerate(sheet_data, start=1):  # start=1 for header offset
            for col_name in sent_row:
                if col_name.endswith("SentenceID") and col_name.startswith(f"{label1}"):
                    sentence_id = sent_row[col_name]
                    sentence_id_locations.setdefault(sheet_name, {})[sentence_id] = row_idx + 1  # Excel row (1-indexed)

    # Add hyperlinks in each sheet to the next sheet's sentence matches
    for label1, label2 in all_pairs:
        #label1 = labels[i]
        #label2 = labels[(i + 1) % len(labels)]
        # Get index of current label
        i = labels.index(label2)
        pageId1 = label1.replace("Page", "")
        pageId2 = label2.replace("Page", "")
        sheet_name = f"{pageId1}vs{pageId2}"[:31]
        worksheet = writer.sheets[sheet_name]
        # Get the DataFrame from stored sheet_data
        df = pd.DataFrame(all_sheet_data[sheet_name])

        for row_idx, row in df.iterrows():
            for col_idx, col_name in enumerate(df.columns):
                if col_name.endswith("SentenceID") and f"Matched{pageId1}vs{pageId2}" in col_name: # Excel Column name used
                    sentence_id = row[col_name]
                    target_sheet = f"{pageId2}vs{labels[(i + 1) % len(labels)].replace('Page','')}"[:31]  # target next sheet # Excel Column name used
                    target_row = sentence_id_locations.get(target_sheet, {}).get(sentence_id)
                    if target_row:
                        cell_ref = f"A{target_row}"
                        formula = f"=HYPERLINK(\"#{target_sheet}!{cell_ref}\", \"{sentence_id}\")"
                        worksheet.write_formula(row_idx + 1, col_idx, formula, link_format)

    writer.close()
    output.seek(0)
    process_times['excel_generation'] = time.time() - t_start

    # Save to file with dynamic name
    t_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Duplicate_Sentence_Report_{timestamp}.xlsx"
    with open(filename, "wb") as f:
        f.write(output.read())
    process_times['file_saving'] = time.time() - t_start

    print(f"✅ Excel Report saved as '{filename}'")
    end_time = time.time()
    print(f"⏱️ Total execution time: {end_time - start_time:.2f} seconds")

      # Print timing information
    print("\n🕒 Process Timing Breakdown:")
    print("=" * 50)
    for process, duration in process_times.items():
        print(f"⌛ {process.replace('_', ' ').title()}: {duration:.2f} seconds")
    print("=" * 50)
    print(f"⏱️ Total execution Breakup time 2: {sum(process_times.values()):.2f} seconds\n")

    return filename

urls = [
    #"https://en.wikipedia.org/wiki/New_York_City",
    #"https://en.wikipedia.org/wiki/Los_Angeles"
    #"https://en.wikipedia.org/wiki/Chicago"
    #"https://en.wikipedia.org/wiki/Cat",
    #"https://en.wikipedia.org/wiki/Dog"
    #"https://en.wikipedia.org/wiki/Fox",
    #"https://en.wikipedia.org/wiki/Bear
    
    "https://example.com",
    "https://www.example.org",
    "https://www.example.net",
    "https://example.com",
    "https://www.example.net"
]

filename = generate_similarity_excel(urls)
