from flask import Flask, render_template, request, jsonify
from pathlib import Path
import math
import pandas as pd

from data_loader import (
    get_dataset_files,
    load_labels
)

from preprocessing import preprocess_dataset

from feature_engineering import (
    create_features,
    FEATURES
)

from anomaly_detection import (
    detect_anomalies_with_scores
)

from evaluation import evaluate_predictions


app = Flask(__name__)


# =========================================================
# MONITORING SESSION
# =========================================================

monitoring_session = {
    "active": False,
    "dataset": None,
    "algorithm": "Isolation Forest",
    "params": None,
    "df": None,
    "predictions": None,
    "scores": None,
    "current_index": 0,
    "metrics": None
}


# =========================================================
# ROUTES (4 Sidebar Pages)
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/algorithm-info")
def algorithm_info():
    return render_template("algorithm_info.html")


@app.route("/anomalies")
def anomalies():
    return render_template("anomalies.html")


@app.route("/performance")
def performance():
    return render_template("performance.html")


# Legacy route support
@app.route("/about")
def about():
    return render_template("index.html")


# =========================================================
# GET DATASETS (Fixed Path Resolution)
# =========================================================

@app.route("/api/datasets")
def get_datasets():
    try:
        files = get_dataset_files()
        datasets = []

        for file_path in files:
            # Safely resolve relative or absolute paths for loading
            datasets.append({
                "name": file_path.name,
                "path": str(file_path.resolve())
            })

        return jsonify({
            "count": len(datasets),
            "datasets": datasets
        })
    except Exception as e:
        app.logger.exception("Failed to fetch datasets")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# START MONITORING
# =========================================================

@app.route("/api/start-monitoring", methods=["POST"])
def start_monitoring():
    try:
        data = request.get_json(silent=True) or {}
        dataset_name = data.get("dataset")
        algorithm = data.get("algorithm", "Isolation Forest")
        params = data.get("params")

        if algorithm not in [
            "Isolation Forest", "LOF", "One-Class SVM", "DBSCAN", "Half-Space Trees", "LSTM Autoencoder"

        ]:
            return jsonify({
                "success": False,
                "error": "Unsupported algorithm selected."
            }), 400

        files = get_dataset_files()
        target_file = next(
            (f for f in files if f.name == dataset_name or str(f.resolve()) == dataset_name or str(f) == dataset_name),
            None
        )

        if not target_file or not target_file.exists():
            return jsonify({
                "success": False,
                "error": "Dataset not found."
            }), 404

        try:
            labels = load_labels()
        except Exception:
            labels = None

        df = preprocess_dataset(Path(target_file), labels)
        df = create_features(df)

        y_true = (
            df["ground_truth"].astype(int).values
            if "ground_truth" in df.columns
            else None
        )

        predictions, scores = detect_anomalies_with_scores(
            df,
            algorithm,
            features=FEATURES,
            params=params
        )

        metrics = (
            evaluate_predictions(y_true, predictions, scores=scores)
            if y_true is not None
            else {}
        )

        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                return value if math.isfinite(value) else None
            if hasattr(value, "item"):
                return clean(value.item())
            return value

        monitoring_session.update({
            "active": True,
            "dataset": target_file.name,
            "algorithm": algorithm,
            "params": params,
            "df": df,
            "predictions": predictions,
            "scores": scores,
            "current_index": 0,
            "metrics": metrics
        })

        return jsonify(clean({
            "success": True,
            "dataset": target_file.name,
            "algorithm": algorithm,
            "params": params,
            "rows": len(df),
            "features": FEATURES,
            "metrics": metrics,
            "predicted_anomalies": int(sum(predictions))
        }))

    except Exception as e:
        app.logger.exception("Failed to start monitoring")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# NEXT OBSERVATION
# =========================================================

@app.route("/api/next-observation")
def next_observation():
    if not monitoring_session["active"] or monitoring_session["df"] is None:
        return jsonify({
            "success": False,
            "error": "No active monitoring session."
        }), 400

    df = monitoring_session["df"]
    predictions = monitoring_session["predictions"]
    scores = monitoring_session["scores"]
    index = monitoring_session["current_index"]

    if index >= len(df):
        monitoring_session["active"] = False
        return jsonify({
            "success": True,
            "completed": True,
            "progress": 100
        })

    row = df.iloc[index]
    prediction = int(predictions[index])
    score = float(scores[index])

    all_scores = monitoring_session["scores"]

    if prediction == 0:
        severity = "Normal"
    else:
        percentile = (all_scores < score).mean() * 100
        if percentile >= 95:
            severity = "High"
        elif percentile >= 85:
            severity = "Medium"
        else:
            severity = "Low"

    progress = ((index + 1) / len(df)) * 100
    monitoring_session["current_index"] += 1

    return jsonify({
        "success": True,
        "completed": False,
        "dataset": monitoring_session["dataset"],
        "observation": index + 1,
        "timestamp": str(row.get("timestamp", index + 1)),
        "value": float(row["value"]),
        "ground_truth": int(row["ground_truth"]) if "ground_truth" in row else 0,
        "anomaly": prediction == 1,
        "anomaly_score": score,
        "severity": severity,
        "progress": progress
    })


# =========================================================
# ANOMALIES API (Dedicated Page Support)
# =========================================================

@app.route("/api/anomalies")
def get_anomalies():
    if monitoring_session["df"] is None or monitoring_session["predictions"] is None:
        return jsonify({
            "success": False,
            "error": "No active detection run found."
        }), 400

    df = monitoring_session["df"]
    predictions = monitoring_session["predictions"]
    scores = monitoring_session["scores"]
    all_scores = scores

    anomaly_list = []
    for idx in range(len(df)):
        if predictions[idx] == 1:
            row = df.iloc[idx]
            score = float(scores[idx])

            percentile = (all_scores < score).mean() * 100
            if percentile >= 95:
                severity = "High"
            elif percentile >= 85:
                severity = "Medium"
            else:
                severity = "Low"

            anomaly_list.append({
                "observation": idx + 1,
                "timestamp": str(row.get("timestamp", idx + 1)),
                "value": float(row["value"]),
                "anomaly_score": score,
                "severity": severity,
                "ground_truth": int(row["ground_truth"]) if "ground_truth" in row else 0
            })

    return jsonify({
        "success": True,
        "dataset": monitoring_session["dataset"],
        "algorithm": monitoring_session.get("algorithm", "Isolation Forest"),
        "total_anomalies": len(anomaly_list),
        "anomalies": anomaly_list
    })


# =========================================================
# PERFORMANCE API
# =========================================================

@app.route("/api/performance")
def get_performance():
    if monitoring_session["metrics"] is None:
        return jsonify({
            "success": False,
            "error": "No monitoring session available."
        }), 400

    metrics = monitoring_session["metrics"]

    return jsonify({
        "success": True,
        "dataset": monitoring_session["dataset"],
        "algorithm": monitoring_session.get("algorithm", "Isolation Forest"),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "f1": metrics.get("f1", 0.0),
        "pr_auc": None if pd.isna(metrics.get("pr_auc")) else float(metrics.get("pr_auc")),
        "true_positive": int(metrics.get("true_positive", 0)),
        "false_positive": int(metrics.get("false_positive", 0)),
        "false_negative": int(metrics.get("false_negative", 0)),
        "true_negative": int(metrics.get("true_negative", 0)),
        "predicted_anomalies": int(metrics.get("predicted_anomalies", 0)),
        "actual_anomalies": int(metrics.get("actual_anomalies", 0))
    })


@app.route("/api/session-status", methods=["GET"])
def session_status():
    if not monitoring_session.get("active", False):
        return jsonify({"active": False})

    df = monitoring_session.get("df")
    current_index = monitoring_session.get("current_index", 0)
    predictions = monitoring_session.get("predictions", [])

    return jsonify({
        "active": True,
        "dataset": monitoring_session.get("dataset"),
        "algorithm": monitoring_session.get("algorithm"),
        "current_index": current_index,
        "total_rows": len(df) if df is not None else 0,
        "metrics": monitoring_session.get("metrics", {}),
        "predicted_anomalies": int(sum(predictions)) if len(predictions) > 0 else 0
    })

@app.route('/research-results')
def research_results():
    return render_template('research_results.html')

from flask import send_from_directory

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename) # Agar folder ka naam 'results' hai

# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)