# scripts/main_baseline.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve


# -----------------------------
# 📊 Visualization Functions
# -----------------------------
def plot_class_distribution(df: pd.DataFrame):
    """Plot class imbalance (fraud vs non-fraud)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Class", data=df, palette="coolwarm", ax=ax)
    ax.set_title("Class Distribution (0 = Non-Fraud, 1 = Fraud)")
    return fig


def plot_amount_distribution(df: pd.DataFrame):
    """Compare transaction amounts for fraud vs non-fraud."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Class", y="Amount", data=df, palette="coolwarm", showfliers=False, ax=ax)
    ax.set_yscale("log")  # fraud amounts can be very skewed
    ax.set_title("Transaction Amounts (Fraud vs Non-Fraud)")
    return fig


def plot_pca_projection(df: pd.DataFrame, n_components: int = 2):
    """Perform PCA and plot fraud vs non-fraud."""
    features = df.drop(columns=["Class", "Time"])
    labels = df["Class"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(features_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        x=components[:, 0], y=components[:, 1],
        hue=labels, palette={0: "blue", 1: "red"}, alpha=0.5, ax=ax
    )
    ax.set_title("PCA Visualization (Fraud vs Non-Fraud)")
    return fig


# -----------------------------
# 🧪 Train & Evaluate Baseline Model
# -----------------------------
def time_split(df: pd.DataFrame, split_ratio: float = 0.8):
    """Split dataset based on Time column (train first X%, test remaining)."""
    df_sorted = df.sort_values("Time")
    split_point = int(len(df_sorted) * split_ratio)
    train = df_sorted.iloc[:split_point]
    test = df_sorted.iloc[split_point:]
    return {"train": train, "test": test}


def train_baseline(train: pd.DataFrame, features: list):
    """Train a simple logistic regression baseline model."""
    X_train = train[features]
    y_train = train["Class"]

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


def evaluate(model, test: pd.DataFrame, features: list, default_threshold: float = 0.5):
    """Evaluate baseline model and return results dictionary."""
    X_test = test[features]
    y_true = test["Class"]

    y_proba = model.predict_proba(X_test)[:, 1]

    # pick threshold = one maximizing F2-like balance between precision and recall
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f2_scores = (5 * prec * rec) / (4 * prec + rec + 1e-8)
    best_idx = f2_scores.argmax()
    best_threshold = thr[best_idx] if best_idx < len(thr) else default_threshold

    return {
        "y_true": y_true.values,
        "y_proba": y_proba,
        "threshold": best_threshold,
    }


# -----------------------------
# Debug Run
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/creditcard.csv")
    print("Data loaded:", df.shape)

    # Show plots
    plot_class_distribution(df).show()
    plot_amount_distribution(df).show()
    plot_pca_projection(df).show()

    # Train baseline
    FEATURES = [c for c in df.columns if c.startswith("V")] + ["Amount"]
    splits = time_split(df)
    model = train_baseline(splits["train"], FEATURES)
    results = evaluate(model, splits["test"], FEATURES)
    print("Evaluation:", results)
