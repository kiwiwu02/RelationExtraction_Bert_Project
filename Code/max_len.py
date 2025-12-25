import json
from transformers import BertTokenizer


# Read the JSON file
def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


# Load BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
tokenizer = BertTokenizer.from_pretrained("/root/models")

# Statistical dataset text length
def get_sentence_lengths(data):
    sentence_lengths = []

    for sample in data:
        sentence = sample["sentence"]
        tokens = tokenizer.tokenize(sentence)  # Get token
        sentence_lengths.append(len(tokens))  # Record the number of tokens

    return sentence_lengths


# Calculate maximum, minimum, and average length
def calculate_statistics(lengths):
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    return max_len, min_len, avg_len


file_path = "./dataset/test.jsonl" 
train_data = read_jsonl(file_path)

# Obtain sentence length
sentence_lengths = get_sentence_lengths(train_data)

# Calculate statistical information
max_len, min_len, avg_len = calculate_statistics(sentence_lengths)

print(f"Maximum sentence length: {max_len}")
print(f"Minimum sentence length: {min_len}")
print(f"Average sentence length: {avg_len:.2f}")
