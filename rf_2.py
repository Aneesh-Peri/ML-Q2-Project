import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

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

print(f"Train Accuracy: {accuracy_score(y_train, train_preds)*100:.2f}%")
print(f"Test Accuracy:  {accuracy_score(y_test, test_preds)*100:.2f}%\n")

print("Classification Metrics (Test Set)")
print(f"Precision: {precision_score(y_test, test_preds):.3f}")
print(f"Recall:    {recall_score(y_test, test_preds):.3f}")
print(f"F1-score:  {f1_score(y_test, test_preds):.3f}\n")

print("Confusion Matrix (Train):")
print(confusion_matrix(y_train, train_preds))

print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, test_preds))

depths = [tree.tree_.max_depth for tree in rf.estimators_]
nodes = [tree.tree_.node_count for tree in rf.estimators_]

print("\nModel Complexity Statistics")
print(f"Average Tree Depth: {sum(depths)/len(depths):.2f}")
print(f"Average Node Count: {sum(nodes)/len(nodes):.2f}")