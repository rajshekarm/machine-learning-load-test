
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load example dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=1
)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, predictions))

# Save model and class names together
joblib.dump(
    {
        "model": model,
        "class_names": iris.target_names.tolist()
    },
    "model.joblib"
)

print("Saved model.joblib")
