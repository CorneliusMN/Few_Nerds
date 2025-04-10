import argparse
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot F1 scores vs Number of Sentences from log file.")
parser.add_argument("file_path", help="Path to the log file")
args = parser.parse_args()
file_path = args.file_path

# Read the file
with open(file_path, 'r') as file:
    lines = file.readlines()

epoch_data = {}
current_epoch = None

for i in range(3):  # specify the number of epochs
    epoch_data[i] = {
        "sentences": [],
        "scores": []
    }

for line in lines:
    if "Epoch number" in line:
        current_epoch = int(line.split()[-1])  # current epoch number
    elif "Number of sentences" in line:
        num_sentences = int(line.split()[-1])  # number of sentences
    elif "f1" in line:
        f1_score = float(line.split()[-1])  # f1 score
        if current_epoch is not None:
            epoch_data[current_epoch]["sentences"].append(num_sentences)
            epoch_data[current_epoch]["scores"].append(f1_score)

# Extract the lists for each epoch
epoch_1_sentences = epoch_data[0]["sentences"]
epoch_1_scores = epoch_data[0]["scores"]
epoch_2_sentences = epoch_data[1]["sentences"]
epoch_2_scores = epoch_data[1]["scores"]
epoch_3_sentences = epoch_data[2]["sentences"]
epoch_3_scores = epoch_data[2]["scores"]

print(len(epoch_1_sentences) == len(epoch_1_scores))
print(len(epoch_2_sentences) == len(epoch_2_scores))
print(len(epoch_3_sentences) == len(epoch_3_scores))

### PLOT ALL EPOCHS

epochs = [1, 2, 3]
sentences_data = [epoch_1_sentences, epoch_2_sentences, epoch_3_sentences]
scores_data = [epoch_1_scores, epoch_2_scores, epoch_3_scores]

plt.figure(figsize=(10, 6))

for epoch, sentences, scores in zip(epochs, sentences_data, scores_data):
    plt.plot(sentences, scores, marker="o", linestyle="-", label=f"Epoch {epoch}")

plt.title("F1 Score vs Number of Sentences")
plt.xlabel("Number of Sentences")
plt.ylabel("F1 Score")
plt.grid(True)
plt.legend()
plt.show()

### PLOT SINGLE EPOCH

plt.figure(figsize=(8, 6))
plt.plot(epoch_1_sentences, epoch_1_scores, marker="o", linestyle="-", label="Epoch 1")
plt.title("F1 Score vs Number of Sentences")
plt.xlabel("Number of Sentences")
plt.ylabel("F1 Score")
plt.grid(True)
plt.legend()
plt.show()
