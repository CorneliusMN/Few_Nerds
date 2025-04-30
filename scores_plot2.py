import argparse
import re
import matplotlib.pyplot as plt

def file_to_lists(path, labels):
    """
    """
    with open(path, "r") as f:
        text = f.read()
    text = text.replace("f1:", "f1 =")
    blocks = re.split(r'(?=Number of sentences:)', text)
    sentences = []
    total_f1 = []
    indiv_f1 = {label: [] for label in labels}

    sent_pattern = re.compile(r'Number of sentences:\s*(\d+)')
    total_pattern = re.compile(r'f1\s*=\s*([0-9.eE+\-]+)')
    label_patterns = {
        label: re.compile(
            rf'Label:\s*{re.escape(label)}\s*\|\s*TP:\s*(\d+)\s*\|\s*FP:\s*(\d+)\s*\|\s*TN:\s*(\d+)\s*\|\s*FN:\s*(\d+)'
        )
        for label in labels
    }

    for block in blocks:
        m_sent = sent_pattern.search(block)
        m_total = total_pattern.search(block)
        if m_sent and m_total:
            N = int(m_sent.group(1))
            ft = float(m_total.group(1))
            sentences.append(N)
            total_f1.append(ft)
            for label, pat in label_patterns.items():
                m_label = pat.search(block)
                if m_label:
                    TP, FP, _, FN = map(int, m_label.groups())
                    # Compute precision and recall
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                    # F1 = 2 * (precision * recall) / (precision + recall)
                    fi = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                else:
                    fi = None
                indiv_f1[label].append(fi)
    return sentences, total_f1, indiv_f1

def main():
    parser = argparse.ArgumentParser(description = "Plot total and per-label F1 vs number of sentences")
    parser.add_argument("--file", "-f", required = True, help = "Path to log file")
    parser.add_argument("--labels", "-l", required = True, help = "Comma-separated list of labels to plot (e.g. 'person-actor,product-car')")
    args = parser.parse_args()
    labels = [lbl.strip() for lbl in args.labels.split(",")]
    sents, total_scores, indiv_scores = file_to_lists(args.file, labels)
    # Print the extracted lists
    print("Number of sentences:", sents)
    print("Total F1 scores:", total_scores)
    for lbl in labels:
        print(f"Individual F1 scores for '{lbl}':", indiv_scores[lbl])
    plt.figure(figsize = (10, 6))
    plt.plot(sents, total_scores, linestyle = "--", alpha = 0.5, label = "Total F1")
    for lbl in labels:
        plt.plot(sents, indiv_scores[lbl], linestyle = "-", label = f"F1: {lbl}")
    plt.title("Total and Per-Label F1 vs Number of Sentences")
    plt.xlabel("Number of Sentences")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# python scores_plot2.py --file product_1_10_s.txt --labels "product-car,product-weapon"