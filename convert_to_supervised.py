import json
import os
import argparse

def process_json_file(filename):
    '''
    takes as input a json file consisting of a dictionary with the keys support and query.
    support and query should contain as key a dictonary containing sentence and label
    return output in ConLL format'''
    support_sentences = []
    query_sentences = []
    
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue # skip empty lines
            data = json.loads(line)
            if "support" in data:
                words_list = data["support"].get("word", [])
                if "label" in data["support"]:
                    labels_list = data["support"].get("label", [])

                else:
                    labels_list = [["O"] * len(words) for words in words_list]
                for words, labels in zip(words_list, labels_list):
                    support_sentences.append((words, labels))
            if "query" in data:
                words_list = data["query"].get("word", [])
                labels_list = data["query"].get("label", [])
                for words, labels in zip(words_list, labels_list):
                    query_sentences.append((words, labels))
    return support_sentences, query_sentences

def save_sentences(sentences, output_filename):
    with open(output_filename, "w", encoding = "utf-8") as f:
        for words, bio_labels in sentences:
            for token, label in zip(words, bio_labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")

def main():
    '''
    Give as input the path to a json file with a dictionary containing the keys support and query. Inside these should be dictionaries
    containing sentence and label'''
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a JSONL file and save support and query sentences.")
    parser.add_argument('input_filename', help="The input JSONL file to process.")

    # Parse the arguments
    args = parser.parse_args()

    input_filename = args.input_filename

    support_output_filename = "support.txt"
    query_output_filename = "query.txt"
    if not os.path.exists(input_filename):
        print("File not found:", input_filename)
    else:
        support_sentences, query_sentences = process_json_file(input_filename)
        save_sentences(support_sentences, support_output_filename)
        save_sentences(query_sentences, query_output_filename)
        print(f"Saved {len(support_sentences)} support sentences with BIO labels to {support_output_filename}")
        print(f"Saved {len(query_sentences)} query sentences with BIO labels to {query_output_filename}")

if __name__ == "__main__":
    main()