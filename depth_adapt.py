import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======================================================
# Depth-Adaptive Pre-Pruned Decision Tree
# ======================================================
class DepthAdaptiveTree:
    def __init__(
        self,
        epsilon_0=0.01,
        alpha=0.15,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features=None
    ):
        self.epsilon_0 = epsilon_0
        self.alpha = alpha
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None

    def entropy(self, y):
        counts = np.bincount(y)
        probs = counts / np.sum(counts)
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def information_gain(self, y, y_left, y_right):
        H_parent = self.entropy(y)
        n = len(y)
        return H_parent - (
            len(y_left) / n * self.entropy(y_left) +
            len(y_right) / n * self.entropy(y_right)
        )

    def best_split(self, X, y):
        best_ig = 0
        best_feature = None
        best_threshold = None

        # -------- FEATURE SUBSAMPLING (KEY FIX) --------
        features = (
            np.random.choice(X.shape[1], self.max_features, replace=False)
            if self.max_features else range(X.shape[1])
        )

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = ~left_mask

                if (
                    np.sum(left_mask) < self.min_samples_leaf or
                    np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                ig = self.information_gain(
                    y,
                    y[left_mask],
                    y[right_mask]
                )

                if ig > best_ig:
                    best_ig = ig
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold, best_ig

    def epsilon(self, depth):
        return self.epsilon_0 * np.exp(-self.alpha * depth)

    def build(self, X, y, depth):
        # Pure node
        if len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        # Depth constraint
        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]

        # Sample constraint
        if len(y) < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]

        feature, threshold, ig = self.best_split(X, y)

        # Depth-adaptive pre-pruning
        if feature is None or ig < self.epsilon(depth):
            return Counter(y).most_common(1)[0][0]

        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build(X[left_idx], y[left_idx], depth + 1),
            "right": self.build(X[right_idx], y[right_idx], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self.build(X, y, depth=0)

    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self.predict_one(x, node["left"])
        return self.predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

    # ---------------- Tree Statistics ----------------
    def count_nodes(self, node=None):
        if node is None:
            node = self.tree
        if not isinstance(node, dict):
            return 1
        return 1 + self.count_nodes(node["left"]) + self.count_nodes(node["right"])

    def tree_depth(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if not isinstance(node, dict):
            return depth
        return max(
            self.tree_depth(node["left"], depth + 1),
            self.tree_depth(node["right"], depth + 1)
        )
class DepthAdaptiveRF:
    def __init__(
        self,
        n_estimators=75,
        epsilon_0=0.01,
        alpha=0.15,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.epsilon_0 = epsilon_0
        self.alpha = alpha
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        n = len(X)

        max_features = int(np.sqrt(X.shape[1]))

        for i in range(self.n_estimators):
            idx = np.random.choice(n, n, replace=True)
            tree = DepthAdaptiveTree(
                epsilon_0=self.epsilon_0,
                alpha=self.alpha,
                max_features=max_features
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(
            lambda x: np.bincount(x, minlength=2).argmax(),
            axis=0,
            arr=preds
        )

    # ---------------- Forest Statistics ----------------
    def average_node_count(self):
        return np.mean([tree.count_nodes() for tree in self.trees])

    def average_tree_depth(self):
        return np.mean([tree.tree_depth() for tree in self.trees])


# ======================================================
# Load Data
# ======================================================
train_df = pd.read_csv("heart_disease_train.csv")
test_df = pd.read_csv("heart_disease_test.csv")

target_col = "target"

X_train = train_df.drop(columns=[target_col]).values
y_train = train_df[target_col].values

X_test = test_df.drop(columns=[target_col]).values
y_test = test_df[target_col].values


model = DepthAdaptiveRF(
    n_estimators=75,
    epsilon_0=0.09,
    alpha=0.15
)

model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Depth-Adaptive Pre-Pruned Random Forest (Fixed & Tuned)\n")

print(f"Train Accuracy: {accuracy_score(y_train, train_preds) * 100:.2f}%")
print(f"Test Accuracy:  {accuracy_score(y_test, test_preds) * 100:.2f}%\n")

print("Classification Metrics (Test Set)")
print(f"Precision: {precision_score(y_test, test_preds):.3f}")
print(f"Recall:    {recall_score(y_test, test_preds):.3f}")
print(f"F1-score:  {f1_score(y_test, test_preds):.3f}\n")

print("Model Complexity Statistics")
print(f"Average Node Count: {model.average_node_count():.2f}")
print(f"Average Tree Depth: {model.average_tree_depth():.2f}")