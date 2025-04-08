import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

def decision_tree_classifier(X, label, X_test, y_test):
    depth = range(1, 20)
    training_errors = []
    test_errors = []
    for i in depth:
        clf = DecisionTreeClassifier(max_depth=i, random_state=42)
        clf.fit(X, label)
        training_errors.append(1 - clf.score(X, label))
        test_errors.append(1 - clf.score(X_test, y_test))
        y_pred = clf.predict(X_test)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, cmap='viridis', marker='o')
        legend1 = ax.legend(*scatter.legend_elements(), title="Predicted")
        ax.add_artist(legend1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f"Decision Tree (Test) Depth:{i}")
        plt.savefig(f"Decision Tree Depth {i}.png")
        plt.close()
    return training_errors, test_errors

def adboost_decision_tree_classifier(X, label, X_test, y_test):
    ada_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=100,
        learning_rate=1,
        algorithm='SAMME',
        random_state=42
    )
    ada_clf.fit(X, label)
    test_error = 1 - ada_clf.score(X_test, y_test)
    y_pred = ada_clf.predict(X_test)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, cmap='viridis', marker='o')
    legend1 = ax.legend(*scatter.legend_elements(), title="Predicted")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Ada Boost (Test)")
    plt.savefig("Decision Tree.png")
    plt.close()

    return test_error

X, label = make_moons_3d(n_samples=1000, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)
training_errors, test_errors = decision_tree_classifier(X, label, X_test, y_test)
test_error = adboost_decision_tree_classifier(X, label, X_test, y_test)

print(test_error)

depth = range(1, 20)
plt.figure(figsize=(8, 4))
plt.plot(depth, training_errors, label='Training Error', marker='o')
plt.plot(depth, test_errors, label='Test Error', marker='s')
plt.xlabel('Depth of Decision Tree')
plt.ylabel('Error')
plt.title('Decision Tree Depth vs. Training and Test Errors on Wine Dataset')
plt.legend()
plt.grid(True)
plt.savefig("Decision Tree Training and Test errors.png")
plt.close()