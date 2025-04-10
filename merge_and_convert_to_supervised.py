import json
import os
import argparse

def process_json_lines(file1, file2):
    """
    Processes two JSONL files line by line.
    - Merges "support" data from both files.
    - Extracts "query" data only from the second file.
    """
    support_sentences = []
    query_sentences = []
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        len1, len2 = len(lines1), len(lines2)
        for i in range(len1):
            line1 = lines1[i].strip()
            if not line1:
                continue
            data1 = json.loads(line1)
            support_list = []
            # Extract "support" from the first file
            if "support" in data1:
                words_list = data1["support"].get("word", [])
                labels_list = data1["support"].get("label", [[]] * len(words_list))
                support_list.extend(zip(words_list, labels_list))
            # If there is a corresponding line in the second file, merge its "support" and extract "query"
            if i < len2:
                line2 = lines2[i].strip()
                if line2:
                    data2 = json.loads(line2)
                    if "support" in data2:
                        words_list = data2["support"].get("word", [])
                        labels_list = data2["support"].get("label", [[]] * len(words_list))
                        support_list.extend(zip(words_list, labels_list))
                    if "query" in data2:
                        words_list = data2["query"].get("word", [])
                        labels_list = data2["query"].get("label", [[]] * len(words_list))
                        query_sentences.extend(zip(words_list, labels_list))
            support_sentences.extend(support_list)
    return support_sentences, query_sentences

def save_sentences(sentences, output_filename):
    """Saves sentences to a file in BIO format."""
    with open(output_filename, "w", encoding="utf-8") as f:
        for words, bio_labels in sentences:
            for token, label in zip(words, bio_labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")

def main():
    """
    Accepts two JSONL input files and processes them as described.
    """
    parser = argparse.ArgumentParser(description="Process two JSONL files and merge support data while extracting query data.")
    parser.add_argument('input_file1', help="First input JSONL file.")
    parser.add_argument('input_file2', help="Second input JSONL file.")
    args = parser.parse_args()
    file1, file2 = args.input_file1, args.input_file2
    if not os.path.exists(file1) or not os.path.exists(file2):
        print("One or both input files not found.")
        return
    support_output_filename = "train.txt"
    query_output_filename = "dev.txt"
    support_sentences, query_sentences = process_json_lines(file1, file2)
    save_sentences(support_sentences, support_output_filename)
    save_sentences(query_sentences, query_output_filename)
    print(f"Saved {len(support_sentences)} support sentences with BIO labels to {support_output_filename}")
    print(f"Saved {len(query_sentences)} query sentences with BIO labels to {query_output_filename}")

if __name__ == "__main__":
    main()

# python3 merge_and_convert_to_supervised.py train_art_2_1.jsonl test_art_2_1.jsonl

# VERY IMPORTANT to give the train first and test second