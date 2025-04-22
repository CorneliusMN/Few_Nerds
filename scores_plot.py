import argparse
import os
import matplotlib.pyplot as plt

def file_to_lists(path):
    """
    Reads a log file and returns two lists:
    - sentences: number of sentences values
    - scores: f1 values
    """
    # sentences, scores = [], []
    # current_num = None
    # with open(path, "r") as f:
    #     for line in f:
    #         if "Number of sentences" in line:
    #             current_num = int(line.split()[-1])
    #         elif "f1" in line and current_num is not None:
    #             sentences.append(current_num)
    #             scores.append(float(line.split()[-1]))
    # return sentences, scores
    sentences, scores = [], []
    current_num = None
    with open(path, "r") as f:
        for line in f:
            if "Number of sentences" in line:
                current_num = int(line.split()[-1])
            elif "f1" in line and current_num is not None:
                # if sentence count has decreased, stop processing
                if sentences and current_num < sentences[-1]:
                    break
                sentences.append(current_num)
                scores.append(float(line.split()[-1]))
    return sentences, scores

def filter_to_common_range(s1, f1, s2, f2):
    """
    Returns filtered s1, f1, s2, f2 and the x-axis limit
    """
    max1 = max(s1) if s1 else 0
    max2 = max(s2) if s2 else 0
    # x_limit = min(max1, max2)
    x_limit = 2000

    def trim(s, f):
        pairs = [(si, fi) for si, fi in zip(s, f) if si <= x_limit]
        if not pairs:
            return [], []
        s_trim, f_trim = zip(*pairs)
        return list(s_trim), list(f_trim)

    s1_f, f1_f = trim(s1, f1)
    s2_f, f2_f = trim(s2, f2)
    return s1_f, f1_f, s2_f, f2_f, x_limit

def main():
    parser = argparse.ArgumentParser(description = "Plot")
    parser.add_argument("--file1", "-1", required = True, help = "path to first log file (.txt)")
    parser.add_argument("--file2", "-2", required = True, help = "path to second log file (.txt)")
    args = parser.parse_args()

    s1, f1 = file_to_lists(args.file1)
    s2, f2 = file_to_lists(args.file2)

    # print(f"{os.path.basename(args.file1)} sentences: {s1}")
    # print(f"{os.path.basename(args.file1)} f1 scores:  {f1}")
    # print(f"{os.path.basename(args.file2)} sentences: {s2}")
    # print(f"{os.path.basename(args.file2)} f1 scores:  {f2}")

    # filter to common x-range and plot
    s1_f, f1_f, s2_f, f2_f, x_limit = filter_to_common_range(s1, f1, s2, f2)

    plt.figure(figsize=(10, 6))
    plt.plot(s1_f, f1_f, linestyle = '-', label = os.path.basename(args.file1))
    plt.plot(s2_f, f2_f, linestyle = '-', label = os.path.basename(args.file2))

    plt.title("F1 Score vs Number of Sentences")
    plt.xlabel("Number of Sentences")
    plt.ylabel("F1 Score")
    plt.xlim(0, x_limit)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# python scores_plot.py --file1 file_name_1.txt --file2 file_name_2.txt