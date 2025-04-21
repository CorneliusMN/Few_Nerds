# import argparse
# import matplotlib.pyplot as plt

# # Parse command-line arguments
# parser = argparse.ArgumentParser(description = "Plot F1 scores vs Number of Sentences from log file.")
# parser.add_argument("file_path", help = "Path to the log file")
# args = parser.parse_args()
# file_path = args.file_path

# # Read the file
# with open(file_path, 'r') as file:
#     lines = file.readlines()

# epoch_data = {}
# current_epoch = None

# for i in range(3):  # specify the number of epochs
#     epoch_data[i] = {
#         "sentences": [],
#         "scores": []
#     }

# for line in lines:
#     if "Epoch number" in line:
#         current_epoch = int(line.split()[-1])  # current epoch number
#     elif "Number of sentences" in line:
#         num_sentences = int(line.split()[-1])  # number of sentences
#     elif "f1" in line:
#         f1_score = float(line.split()[-1])  # f1 score
#         if current_epoch is not None:
#             epoch_data[current_epoch]["sentences"].append(num_sentences)
#             epoch_data[current_epoch]["scores"].append(f1_score)

# # Extract the lists for each epoch
# epoch_1_sentences = epoch_data[0]["sentences"]
# epoch_1_scores = epoch_data[0]["scores"]
# epoch_2_sentences = epoch_data[1]["sentences"]
# epoch_2_scores = epoch_data[1]["scores"]
# epoch_3_sentences = epoch_data[2]["sentences"]
# epoch_3_scores = epoch_data[2]["scores"]

# print(len(epoch_1_sentences) == len(epoch_1_scores))
# print(len(epoch_2_sentences) == len(epoch_2_scores))
# print(len(epoch_3_sentences) == len(epoch_3_scores))

# ### PLOT ALL EPOCHS

# epochs = [1, 2, 3]
# sentences_data = [epoch_1_sentences, epoch_2_sentences, epoch_3_sentences]
# scores_data = [epoch_1_scores, epoch_2_scores, epoch_3_scores]

# plt.figure(figsize=(10, 6))

# for epoch, sentences, scores in zip(epochs, sentences_data, scores_data):
#     plt.plot(sentences, scores, marker="o", linestyle="-", label=f"Epoch {epoch}")

# plt.title("F1 Score vs Number of Sentences")
# plt.xlabel("Number of Sentences")
# plt.ylabel("F1 Score")
# plt.grid(True)
# plt.legend()
# plt.show()

# ### PLOT SINGLE EPOCH

# plt.figure(figsize=(8, 6))
# plt.plot(epoch_1_sentences, epoch_1_scores, marker="o", linestyle="-", label="Epoch 1")
# plt.title("F1 Score vs Number of Sentences")
# plt.xlabel("Number of Sentences")
# plt.ylabel("F1 Score")
# plt.grid(True)
# plt.legend()
# plt.show()


import argparse
import os
import matplotlib.pyplot as plt

def file_to_lists(path):
    """
    Reads a log file and returns two lists:
    - sentences: number of sentences values
    - scores: f1 values
    """
    sentences, scores = [], []
    current_num = None
    with open(path, "r") as f:
        for line in f:
            if "Number of sentences" in line:
                current_num = int(line.split()[-1])
            elif "f1" in line and current_num is not None:
                sentences.append(current_num)
                scores.append(float(line.split()[-1]))
    return sentences, scores

def filter_to_common_range(s1, f1, s2, f2):
    """
    Returns filtered s1, f1, s2, f2 and the x-axis limit
    """
    max1 = max(s1) if s1 else 0
    max2 = max(s2) if s2 else 0
    x_limit = min(max1, max2)

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
    plt.plot(s1_f, f1_f, marker = 'o', linestyle = '-', label = os.path.basename(args.file1))
    plt.plot(s2_f, f2_f, marker = 'o', linestyle = '-', label = os.path.basename(args.file2))

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