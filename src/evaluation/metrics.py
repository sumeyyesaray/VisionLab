from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def macro_f1_score(y_true: list[int], y_pred: list[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_report(y_true: list[int], y_pred: list[int], idx_to_label: dict[int, str]) -> list[dict]:
    """Precision/recall/f1/support for every class, in label-index order."""
    num_classes = len(idx_to_label)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0
    )
    return [
        {
            "label": idx_to_label[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    ]


def worst_classes(per_class: list[dict], top_k: int = 10) -> list[dict]:
    """The `top_k` classes with the lowest F1 — where the model struggles most."""
    return sorted(per_class, key=lambda row: row["f1"])[:top_k]


def top_confused_pairs(
    y_true: list[int], y_pred: list[int], idx_to_label: dict[int, str], top_k: int = 10
) -> list[dict]:
    """The `top_k` (true, predicted) off-diagonal cells with the most misclassifications."""
    num_classes = len(idx_to_label)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    pairs = []
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                pairs.append(
                    {
                        "true_label": idx_to_label[true_idx],
                        "predicted_label": idx_to_label[pred_idx],
                        "count": int(cm[true_idx, pred_idx]),
                    }
                )
    pairs.sort(key=lambda row: -row["count"])
    return pairs[:top_k]


def build_evaluation_report(
    y_true: list[int], y_pred: list[int], idx_to_label: dict[int, str], top_k: int = 10
) -> dict:
    per_class = per_class_report(y_true, y_pred, idx_to_label)
    return {
        "macro_f1": macro_f1_score(y_true, y_pred),
        "worst_classes": worst_classes(per_class, top_k=top_k),
        "top_confused_pairs": top_confused_pairs(y_true, y_pred, idx_to_label, top_k=top_k),
    }
