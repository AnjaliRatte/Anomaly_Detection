import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


def evaluate_predictions(y_true, y_pred, scores=None):
    """
    Evaluate anomaly detection predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
        1 = anomaly, 0 = normal

    y_pred : array-like
        Predicted labels.
        1 = anomaly, 0 = normal

    scores : array-like, optional
        Continuous anomaly scores.
        Required for PR-AUC when available.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # PR-AUC
    if scores is not None and len(np.unique(y_true)) > 1:
        pr_auc = average_precision_score(
            y_true,
            scores
        )
    else:
        pr_auc = np.nan

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_anomalies": int(y_pred.sum()),
        "actual_anomalies": int(y_true.sum())
    }


if __name__ == "__main__":

    # Small test to verify that the evaluation module works.
    y_true = [0, 0, 0, 1, 1, 0, 1, 0]
    y_pred = [0, 0, 1, 1, 0, 0, 1, 0]

    results = evaluate_predictions(y_true, y_pred)

    print("Evaluation Test:")
    print("-----------------")

    for metric, value in results.items():
        print(f"{metric}: {value}")