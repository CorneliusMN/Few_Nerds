import os
import json
import argparse
import random
import numpy as np
import torch

from util.data_loader import Sample, FewShotNERDatasetWithRandomSampling
from util.fewshotsampler import FewshotSampler

# Dummy tokenizer that does nothing
class DummyTokenizer:
    def tokenize(self, word):
        return [word]
    def convert_tokens_to_ids(self, tokens):
        return tokens

def read_allowed_labels(labels_file):
    with open(labels_file, "r", encoding = "utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

def custom_populate(dataset, idx_list):
    """
    """
    data = {"word": [], "label": []}
    for idx in idx_list:
        sample = dataset.samples[idx]
        words = list(sample.words)
        labels = list(sample.normalized_tags)
        data["word"].append(words)
        data["label"].append(labels)
    return data

def create_episode(dataset, sampler):
    """
    """
    target_classes, support_idx, query_idx = sampler.__next__()
    support_set = custom_populate(dataset, support_idx)
    query_set = custom_populate(dataset, query_idx)
    episode = {"support": support_set, "query": query_set, "types": target_classes}
    return episode

def main():
    parser = argparse.ArgumentParser(
        description = "Create few-shot NER episodes with location labels and equal support/query sizes (without mask, sentence_num, label2tag, or index)."
    )
    parser.add_argument("--data_file", type = str, required = True,
                        help = "Path to the raw NER training file")
    parser.add_argument("--labels_file", type = str, required = True,
                        help = "Path to the file listing allowed labels")
    parser.add_argument("--N", type = int, required = True, help = "N-way: number of entity types per episode")
    parser.add_argument("--K", type = int, required = True, help = "K-shot: number of support samples per class")
    parser.add_argument("--num_episodes", type = int, default = 100,
                        help = "Total number of episodes to generate")
    parser.add_argument("--output_file", type = str, required = True,
                        help="Path to the output jsonl file")
    parser.add_argument("--max_length", type = int, default = 100,
                        help = "Maximum sequence length (kept for compatibility)") # Not really used here
    parser.add_argument("--seed", type = int, default = 0,
                        help = "Random seed for reproducibility")
    #parser.add_argument("--label_prefix", type = str, default = "person-",
    #                help = "Prefix to filter allowed labels (e.g., 'person-')")
    parser.add_argument("--label_prefixes", nargs="+", type=str, default=["person-"],
                    help="List of prefixes to filter allowed labels (e.g., 'person-' 'location-')")

    args = parser.parse_args()

    Q = args.K

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    allowed_labels = read_allowed_labels(args.labels_file)
    #allowed_labels = [label for label in allowed_labels if label.startswith(args.label_prefix)]
    #print("Allowed labels (filtered with prefix '{}') from {}:".format(args.label_prefix, args.labels_file))
    allowed_labels = [label for label in allowed_labels 
                  if any(label.startswith(prefix) for prefix in args.label_prefixes)]
    print("Allowed labels (filtered with prefix '{}') from {}:".format(args.label_prefixes, args.labels_file))
    print(allowed_labels)

    # Use the DummyTokenizer instead of a real tokenizer
    tokenizer = DummyTokenizer()

    dataset = FewShotNERDatasetWithRandomSampling(
        filepath = args.data_file,
        tokenizer = tokenizer,
        N = args.N, K = args.K, Q = Q,
        max_length = args.max_length,
        ignore_label_id = -1
    )

    original_classes = set(dataset.classes)
    allowed_set = set(allowed_labels)
    new_classes = list(original_classes.intersection(allowed_set))
    if len(new_classes) < args.N:
        raise ValueError("After filtering, only {} classes remain but N={} was requested."
                         .format(len(new_classes), args.N))
    dataset.classes = new_classes

    sampler = FewshotSampler(
        N = args.N,
        K = args.K,
        Q = Q,
        samples = dataset.samples,
        classes = dataset.classes,
        random_state = args.seed
    )

    with open(args.output_file, "w", encoding = "utf-8") as fout:
        for i in range(args.num_episodes):
            try:
                episode = create_episode(dataset, sampler)
                fout.write(json.dumps(episode) + "\n")
                if (i + 1) % 10 == 0:
                    print("Created {} episodes".format(i + 1))
            except Exception as e:
                print("Error creating episode {}: {}".format(i, e))
                continue

    print("Episodes saved to", args.output_file)

if __name__ == "__main__":
    main()

# Example of how to run:
# python3 create_episodes.py --data_file data/inter/all.txt --labels_file data/labels.txt --N 2 --K 2 --num_episodes 20 --max_length 100 --output_file episodes_trial.jsonl