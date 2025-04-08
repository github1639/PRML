import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import svm

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

def SVM(X, label, X_test, y_test):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    classifiers = [svm.SVC(kernel=k, gamma='auto').fit(X, label) for k in kernels]
    accuracies = [clf.score(X_test, y_test) for clf in classifiers]

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100),
                         np.linspace(z_min, z_max, 100))

    fig = plt.figure(figsize=(12, 12))

    axes = [fig.add_subplot(2, 2, i + 1, projection='3d') for i in range(4)]

    for i, (clf, ax, kernel) in enumerate(zip(classifiers, axes, kernels)):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
        Z = Z.reshape(xx.shape)

        y_pred = clf.predict(X_test)
        ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred, cmap='viridis', marker='o')
        # ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(), c=Z, cmap='viridis', marker='.')
        ax.set_title(f"{kernel.capitalize()} Kernel - Accuracy: {accuracies[i]:.2f}")

    plt.tight_layout()
    plt.savefig("SVM (Test).png")
    plt.close()

X, label = make_moons_3d(n_samples=1000, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)

SVM(X, label, X_test, y_test)