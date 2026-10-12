from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree

# Load the dataset from the same folder as this script, independent of the
# terminal's current working directory.
dataset_path = Path(__file__).resolve().parent / "Churn_Modelling.csv"
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(exist_ok=True)
data = pd.read_csv(dataset_path)

# Display data
print(data.head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove missing rows
data = data.dropna()

# Remove ID columns because they are not useful for prediction
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode categorical variables
encoder = LabelEncoder()

for column in data.select_dtypes(include="object").columns:
    data[column] = encoder.fit_transform(data[column])

# Separate features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Decision Tree model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance")
print("----------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm
).plot()

plt.title("Confusion Matrix")
plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Visualize Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Exited", "Exited"],
    filled=True
)

plt.title("Decision Tree for Customer Exit Prediction")
plt.savefig(output_dir / "decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Visualize and save feature importance
plt.figure(figsize=(10, 6))
importance.sort_values().plot(kind="barh", color="steelblue")
plt.title("Feature Importance for Customer Churn Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print(f"\nGraphs saved in: {output_dir}")