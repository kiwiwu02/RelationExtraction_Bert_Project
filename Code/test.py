# test.py
import torch
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, classification_report
from data import RelationshipDataset
from sklearn.metrics import confusion_matrix
import numpy as np

def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def evaluate_model(model_path, test_data, text_build_type):
    # configuration parameters
    MAX_LEN = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading models and tokenizers
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    # Prepare test data
    test_dataset = RelationshipDataset(test_data, tokenizer, MAX_LEN, text_build_type)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # Evaluation mode
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # # Calculate micro indicators
    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     true_labels, predictions, average="micro"
    # )

    # Get all categories（0-9）
    classes = np.unique(true_labels)
    n_classes = len(classes)

    # Calculate confusion matrix (rows: true labels, columns: predicted labels)
    cm = confusion_matrix(true_labels, predictions, labels=classes)

    # Initialize the storage of various category indicators
    class_tp = []
    class_fp = []
    class_fn = []
    class_acc = []

    for i in range(n_classes):
        # Calculate the TP of the current category (true example)
        tp = cm[i, i]
        # Calculate the FP of the current category (false positive example: other categories are predicted as the current category)
        fp = cm[:, i].sum() - tp
        # Calculate the FN of the current category (false negative example: predicting the current category as another category)
        fn = cm[i, :].sum() - tp
        # Calculate the accuracy of the current category (as a weight)
        acc = tp / cm[i, :].sum() if cm[i, :].sum() > 0 else 0.0

        class_tp.append(tp)
        class_fp.append(fp)
        class_fn.append(fn)
        class_acc.append(acc)

    # Normalize weights
    class_acc = np.array(class_acc)
    weights = class_acc / class_acc.sum()

    # Weighted calculation of total TP, FP, FN
    weighted_tp = sum(tp * w for tp, w in zip(class_tp, weights))
    weighted_fp = sum(fp * w for fp, w in zip(class_fp, weights))
    weighted_fn = sum(fn * w for fn, w in zip(class_fn, weights))

    # Calculate the weighted micro index
    precision = weighted_tp / (weighted_tp + weighted_fp + 1e-6)  
    recall = weighted_tp / (weighted_tp + weighted_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    # Calculate Macro Metrics
    print(classification_report(true_labels, predictions, digits=4))

    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    test_data = read_jsonl("./dataset/test.jsonl")

    # label_counts = {}
    # for sample in test_data:
    #     label = sample["r"]
    #     label_counts[label] = label_counts.get(label, 0) + 1
    # print("Distribution of test set categories：", label_counts)  # Check if the quantity of all categories is consistent

    text_build_model = ["basic1", "basic2", "QA", "entity_marked1", "entity_marked2"]

    # Test each model version
    for v in range(5):
        model_path = f"./checkpoint/model{v}"
        results = evaluate_model(model_path, test_data, text_build_model[v])

        print(f"\nResults for model {text_build_model[v]}:")
        print(f"Micro-Precision: {results['precision']:.4f}")
        print(f"Micro-Recall: {results['recall']:.4f}")
        print(f"Micro-F1: {results['f1']:.4f}")


if __name__ == "__main__":
    main()
