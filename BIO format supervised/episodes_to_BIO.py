# # Support and query

# import json
# import os

# def convert_to_bio(words, labels):
#     bio_labels = []
#     for i, label in enumerate(labels):
#         if label == "O":
#             bio_labels.append("O")
#         else:
#             if i == 0 or labels[i-1] != label:
#                 bio_labels.append("B-" + label)
#             else:
#                 bio_labels.append("I-" + label)
#     return bio_labels

# def process_json_file(filename):
#     sentences_list = []
#     with open(filename, "r", encoding = "utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue # skip empty lines
#             data = json.loads(line)
#             # Support sentences first
#             if "support" in data:
#                 words_list = data["support"].get("word", [])
#                 labels_list = data["support"].get("label", [])
#                 for words, labels in zip(words_list, labels_list):
#                     bio_labels = convert_to_bio(words, labels)
#                     sentences_list.append((words, bio_labels))
#             # Then process query sentences
#             if "query" in data:
#                 words_list = data["query"].get("word", [])
#                 labels_list = data["query"].get("label", [])
#                 for words, labels in zip(words_list, labels_list):
#                     bio_labels = convert_to_bio(words, labels)
#                     sentences_list.append((words, bio_labels))
#     return sentences_list

# def save_sentences(sentences, output_filename):
#     with open(output_filename, "w", encoding = "utf-8") as f:
#         for words, bio_labels in sentences:
#             for token, label in zip(words, bio_labels):
#                 f.write(f"{token}\t{label}\n")
#             f.write("\n")

# if __name__ == "__main__":
#     input_filename = "/Users/pietrorebecchi/Documents/SECOND YEAR/FOURTH SEMESTER/NATURAL LANGUAGE PROCESSING AND DEEP LEARNING/Project/Few-NERD/data/episode-data/inter/train_10_5.jsonl"
#     output_filename = "sentences_bio.txt"
#     if not os.path.exists(input_filename):
#         print("File not found:", input_filename)
#     else:
#         sentences = process_json_file(input_filename)
#         save_sentences(sentences, output_filename)
#         print(f"Saved {len(sentences)} sentences (with BIO labels) to {output_filename}")


# Support and query separately

import json
import os

def convert_to_bio(words, labels):
    bio_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            bio_labels.append("O")
        else:
            if i == 0 or labels[i-1] != label:
                bio_labels.append("B-" + label)
            else:
                bio_labels.append("I-" + label)
    return bio_labels

def process_json_file(filename):
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
                # elif "label" in data["support"]:
                #     labels_list = data["support"].get("label", [])
                else:
                    labels_list = [["O"] * len(words) for words in words_list]
                for words, labels in zip(words_list, labels_list):
                    bio_labels = convert_to_bio(words, labels)
                    support_sentences.append((words, bio_labels))
            if "query" in data:
                words_list = data["query"].get("word", [])
                labels_list = data["query"].get("label", [])
                for words, labels in zip(words_list, labels_list):
                    bio_labels = convert_to_bio(words, labels)
                    query_sentences.append((words, bio_labels))
    return support_sentences, query_sentences

def save_sentences(sentences, output_filename):
    with open(output_filename, "w", encoding = "utf-8") as f:
        for words, bio_labels in sentences:
            for token, label in zip(words, bio_labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")

if __name__ == "__main__":
    input_filename = "/Users/pietrorebecchi/Documents/SECOND YEAR/FOURTH SEMESTER/NATURAL LANGUAGE PROCESSING AND DEEP LEARNING/Project/Few-NERD-P/data/episode-data/inter/train_10_5.jsonl"
    support_output_filename = "/Users/pietrorebecchi/Documents/SECOND YEAR/FOURTH SEMESTER/NATURAL LANGUAGE PROCESSING AND DEEP LEARNING/Project/Few-NERD-P/support_sentences_bio.txt"
    query_output_filename = "/Users/pietrorebecchi/Documents/SECOND YEAR/FOURTH SEMESTER/NATURAL LANGUAGE PROCESSING AND DEEP LEARNING/Project/Few-NERD-P/query_sentences_bio.txt"
    if not os.path.exists(input_filename):
        print("File not found:", input_filename)
    else:
        support_sentences, query_sentences = process_json_file(input_filename)
        save_sentences(support_sentences, support_output_filename)
        save_sentences(query_sentences, query_output_filename)
        print(f"Saved {len(support_sentences)} support sentences with BIO labels to {support_output_filename}")
        print(f"Saved {len(query_sentences)} query sentences with BIO labels to {query_output_filename}")