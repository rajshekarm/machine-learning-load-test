import joblib

# Load the saved object
bundle = joblib.load("model.joblib")

model = bundle["model"]
class_names = bundle["class_names"]

print("Saved object keys:", bundle.keys())
print("Model type:", type(model))
print("Number of trees:", len(model.estimators_))
print("Class names:", class_names)
print("Number of input features:", model.n_features_in_)
print("Classes:", model.classes_)

# Inspect the first decision tree
tree = model.estimators_[0].tree_

print("Nodes in first tree:", tree.node_count)
print("Feature indices:", tree.feature[:10])
print("Thresholds:", tree.threshold[:10])