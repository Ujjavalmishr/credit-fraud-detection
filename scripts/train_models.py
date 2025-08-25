from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, matthews_corrcoef, fbeta_score
)
import numpy as np
import joblib
import pandas as pd
from pathlib import Path


def compute_cost(y_true, y_pred, cost_fp=1, cost_fn=10):
    """Custom cost: FP and FN have different penalties."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return fp * cost_fp + fn * cost_fn


def train_and_save_models(train, test, save_dir="models"):
    X_train, y_train = train.drop("Class", axis=1), train["Class"]
    X_test, y_test = test.drop("Class", axis=1), test["Class"]

    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200),
        "svm": SVC(probability=True)
    }

    Path(save_dir).mkdir(exist_ok=True)

    all_results = {}

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, f"{save_dir}/{name}.pkl")

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cost = compute_cost(y_test, y_pred)

        # Store results
        all_results[name] = {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "F2": f2,
            "MCC": mcc,
            "Confusion Matrix": cm.tolist(),
            "Expected Cost": cost
        }

        print(f"✅ {name} trained | ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, "
              f"F2={f2:.4f}, MCC={mcc:.4f}, Cost={cost}")

    return all_results


if __name__ == "__main__":
    print("📂 Loading train/test data ...")
    df = pd.read_csv("data/creditcard.csv")

    # simple split (you can replace with train/test split logic)
    train = df.sample(frac=0.8, random_state=42)
    test = df.drop(train.index)

    print("🚀 Training models ...")
    results = train_and_save_models(train, test, save_dir="models")

    print("\n📊 Training Results:")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1']:.4f}")
        print(f"  ROC-AUC  : {metrics['ROC-AUC']:.4f}")
        print(f"  PR-AUC   : {metrics['PR-AUC']:.4f}")
        print(f"  F2 Score : {metrics['F2']:.4f}")
        print(f"  MCC      : {metrics['MCC']:.4f}")
        print(f"  Cost     : {metrics['Expected Cost']}")
