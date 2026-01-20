import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import export_text

train_data = pd.read_csv("heart_disease_train.csv")
test_data = pd.read_csv("heart_disease_test.csv")

target_col = train_data.columns[-1]

X_train = train_data.drop(columns=[target_col])
y_train = train_data[target_col]

X_test = test_data.drop(columns=[target_col])
y_test = test_data[target_col]

rf = RandomForestClassifier(
    n_estimators=75,
    criterion="entropy",
    min_impurity_decrease=0.01,
    random_state=42
)

rf.fit(X_train, y_train)

train_preds = rf.predict(X_train)
test_preds = rf.predict(X_test)

print("Baseline RF (Fixed Pre-Pruning)\n")

print(f"Train Accuracy: {accuracy_score(y_train, train_preds) * 100:.2f}%")
print(f"Test Accuracy:  {accuracy_score(y_test, test_preds) * 100:.2f}%\n")

print("Classification Metrics (Test Set)")
print(f"Precision: {precision_score(y_test, test_preds):.3f}")
print(f"Recall:    {recall_score(y_test, test_preds):.3f}")
print(f"F1-score:  {f1_score(y_test, test_preds):.3f}\n")

depths = [tree.tree_.max_depth for tree in rf.estimators_]
nodes = [tree.tree_.node_count for tree in rf.estimators_]

print("\nModel Complexity Statistics")
print(f"Average Tree Depth: {sum(depths) / len(depths):.2f}")
print(f"Average Node Count: {sum(nodes) / len(nodes):.2f}")

print("\nFull Decision Tree (Tree 0 in the Forest):\n")
tree_text = export_text(
    rf.estimators_[0],
    feature_names=list(X_train.columns)
)
print(tree_text)

cm_test = confusion_matrix(y_test, test_preds)

fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(cm_test, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["No Disease", "Disease"])
ax.set_yticklabels(["No Disease", "Disease"])
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix (Test Set)")
for i in range(cm_test.shape[0]):
    for j in range(cm_test.shape[1]):
        color = "white" if cm_test[i,j] > cm_test.max()/2 else "black"
        ax.text(j, i, cm_test[i,j], ha="center", va="center", color=color, fontsize=12)
plt.tight_layout()
plt.show()