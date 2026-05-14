"""
demo.py — End-to-end demo: load model → generate paper → evaluate.

Run:
    python demo.py
"""
import logging

from model import CodeToPaperModel
from evaluate import PaperEvaluator
from config import CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Sample code snippet ────────────────────────────────────────────────────────

SAMPLE_CODE = '''
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_neural_network(X, y, hidden_layers=(100, 50), max_iter=500):
    """
    Train a multi-layer perceptron classifier.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    hidden_layers : tuple  – sizes of hidden layers
    max_iter : int         – maximum training iterations

    Returns
    -------
    model   : fitted MLPClassifier
    scaler  : fitted StandardScaler
    metrics : dict of evaluation metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds   = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "report":   classification_report(y_test, preds),
    }
    return model, scaler, metrics
'''

# ── Reference paper (for evaluation) ──────────────────────────────────────────

SAMPLE_REFERENCE = {
    "abstract": (
        "We present a multi-layer perceptron (MLP) framework for supervised classification. "
        "Our implementation leverages ReLU activations, Adam optimisation, and early stopping "
        "to achieve competitive accuracy while mitigating over-fitting."
    ),
    "introduction": (
        "Artificial neural networks have become the cornerstone of modern machine learning. "
        "MLPs remain highly effective for tabular data. "
        "In this work we describe a clean, reproducible MLP pipeline using scikit-learn."
    ),
    "methodology": (
        "The pipeline begins with an 80/20 stratified train-test split. "
        "Features are standardised using zero-mean unit-variance scaling. "
        "A two-hidden-layer MLP with sizes (100, 50) is trained with Adam at lr=1e-3. "
        "Early stopping monitors the validation loss to prevent over-fitting."
    ),
    "results": (
        "The model achieves high accuracy on the held-out test set. "
        "Classification reports show balanced precision and recall across classes. "
        "Early stopping typically triggers within 200–400 iterations."
    ),
    "conclusion": (
        "We demonstrated a reliable MLP training pipeline. "
        "Future work will explore deeper architectures, dropout regularisation, "
        "and automated hyper-parameter search via Optuna."
    ),
}


# ── Demo ───────────────────────────────────────────────────────────────────────

def demo():
    logger.info("=== CodeLens Research — Demo ===")

    # 1. Load model
    model = CodeToPaperModel(CFG)

    # 2. (Optional) Fine-tune on your own data
    #    Provide a list of dicts: [{"code": "...", "abstract": "...", ...}, ...]
    #
    #    from dataset import CodePaperDataset
    #    dataset = CodePaperDataset(model.tokenizer).from_list(your_data)
    #    model.train(dataset)

    # 3. Generate paper
    logger.info("Generating paper from sample code …")
    paper = model.generate_paper(SAMPLE_CODE)

    # 4. Print formatted output
    formatted = model.format_paper(paper, title="Neural Network Classification Pipeline")
    print("\n" + "=" * 70)
    print(formatted)
    print("=" * 70 + "\n")

    # 5. Evaluate against reference
    evaluator = PaperEvaluator()
    scores = evaluator.evaluate(paper, SAMPLE_REFERENCE)

    print("Evaluation (ROUGE + readability):")
    for section, metrics in scores.items():
        print(f"  [{section}]  {metrics}")

    return paper, scores


if __name__ == "__main__":
    demo()
