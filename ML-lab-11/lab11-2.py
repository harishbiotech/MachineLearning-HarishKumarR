import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math

def load_data():
    df = pd.read_csv("Iris.csv")
    X = df.drop(columns=['Id', 'Species'])
    y = df['Species'].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
    return X, y


def root_entropy(y):
    proportions = y.value_counts(normalize=True).tolist()
    entropy = sum(-p * math.log(p, 2) for p in proportions if p > 0)
    return entropy

def split_points(X):
    splits = {}
    for col in X.columns:
        sorted_vals = X[col].sort_values().unique()
        midpoints = [(sorted_vals[i] + sorted_vals[i+1]) / 2 for i in range(len(sorted_vals)-1)]
        splits[col] = midpoints
    return splits

def information_gain(X_col, y, threshold):
    left_y  = y[X_col <= threshold]
    right_y = y[X_col > threshold]

    n = len(y)
    n_left  = len(left_y)
    n_right = len(right_y)

    if n_left == 0 or n_right == 0:
        return 0

    ig = root_entropy(y) - (n_left/n) * root_entropy(left_y) - (n_right/n) * root_entropy(right_y)
    return ig


def best_split(X, y):
    splits = split_points(X)
    best_ig = 0
    best_col = None
    best_threshold = None

    for col, thresholds in splits.items():
        for threshold in thresholds:
            ig = information_gain(X[col], y, threshold)
            if ig > best_ig:
                best_ig = ig
                best_col = col
                best_threshold = threshold

    return best_col, best_threshold, best_ig

def majority_class(y):
    # Returns the most common class in y
    return y.value_counts().idxmax()


def build_tree(X, y, depth=0, max_depth=5, min_samples=5):
    #  STOP condition 1: all samples are same class
    print(type(y))
    if len(y.unique()) == 1:
        return {"leaf": True, "class": y.iloc[0]}

    #  STOP condition 2: max depth reached
    if depth >= max_depth:
        return {"leaf": True, "class": majority_class(y)}

    # STOP condition 3: too few samples
    if len(y) < min_samples:
        return {"leaf": True, "class": majority_class(y)}

    #  STEP 1: Find the best split
    col, threshold, ig = best_split(X, y)

    #  STOP condition 4: no useful split found
    if ig == 0:
        return {"leaf": True, "class": majority_class(y)}

    #  STEP 2: Split data into left and right
    left_mask = X[col] <= threshold
    right_mask = X[col] > threshold

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    # STEP 3: Recurse on left and right
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth, min_samples)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth, min_samples)

    # Return the decision node
    return {
        "leaf": False,
        "col": col,
        "threshold": threshold,
        "left": left_subtree,  # X[col] <= threshold goes left
        "right": right_subtree  # X[col] >  threshold goes right
    }


def predict_one(tree, row):
    # row is a single sample (pandas Series)

    if tree["leaf"]:
        return tree["class"]  # we reached a leaf, return the class

    # Go left or right based on the split condition
    if row[tree["col"]] <= tree["threshold"]:
        return predict_one(tree["left"], row)
    else:
        return predict_one(tree["right"], row)


def predict(tree, X):
    return X.apply(lambda row: predict_one(tree, row), axis=1)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_NAMES = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
CLASS_COLORS = {0: "#9FE1CB", 1: "#B5D4F4", 2: "#F5C4B3"}
NODE_COLOR = "#D3D1C7"

def get_tree_depth(tree):
    if tree["leaf"]:
        return 0
    return 1 + max(get_tree_depth(tree["left"]), get_tree_depth(tree["right"]))

def plot_node(ax, text, x, y, color, fontsize=8):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color,
                      edgecolor="#5F5E5A", linewidth=0.8))

def plot_tree_recursive(ax, tree, x, y, dx, dy):
    if tree["leaf"]:
        cls = tree["class"]
        label = f"class: {CLASS_NAMES[cls]}"
        plot_node(ax, label, x, y, CLASS_COLORS[cls])
        return

    label = f"{tree['col']} <= {tree['threshold']:.2f}"
    plot_node(ax, label, x, y, NODE_COLOR)

    # Left child
    lx, ly = x - dx, y - dy
    ax.annotate("", xy=(lx, ly + 0.02), xytext=(x, y - 0.02),
                arrowprops=dict(arrowstyle="-|>", color="#5F5E5A", lw=0.8))
    ax.text((x + lx) / 2 - 0.01, (y + ly) / 2, "yes",
            fontsize=7, color="#0F6E56", ha="right")
    plot_tree_recursive(ax, tree["left"],  lx, ly, dx / 2, dy)

    # Right child
    rx, ry = x + dx, y - dy
    ax.annotate("", xy=(rx, ry + 0.02), xytext=(x, y - 0.02),
                arrowprops=dict(arrowstyle="-|>", color="#5F5E5A", lw=0.8))
    ax.text((x + rx) / 2 + 0.01, (y + ry) / 2, "no",
            fontsize=7, color="#C04828", ha="left")
    plot_tree_recursive(ax, tree["right"], rx, ry, dx / 2, dy)

def plot_my_tree(tree):
    depth = get_tree_depth(tree)
    fig_h = max(6, depth * 2.5)
    fig_w = max(10, 2 ** depth * 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("My Decision Tree — Iris Dataset", fontsize=13, pad=12)

    dy = 0.85 / max(depth, 1)
    plot_tree_recursive(ax, tree, x=0.5, y=0.95, dx=0.22, dy=dy)

    # Legend
    patches = [mpatches.Patch(facecolor=CLASS_COLORS[i], edgecolor="#5F5E5A",
                               label=CLASS_NAMES[i]) for i in CLASS_NAMES]
    ax.legend(handles=patches, loc="lower right", fontsize=8, title="Classes")

    plt.tight_layout()
    plt.savefig("my_decision_tree.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Tree saved as my_decision_tree.png")

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the tree
tree = build_tree(X_train, y_train, max_depth=5, min_samples=0)

# Predict
y_pred = predict(tree, X_test)


# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

plot_my_tree(tree)
