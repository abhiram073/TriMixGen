import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        id2label[p] for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100
    ]
    true_labels = [
        id2label[l] for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100
    ]

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average="macro", zero_division=0)
    acc = accuracy_score(true_labels, true_predictions)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "macro_f1": f1
    }
