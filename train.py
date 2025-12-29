import time
import numpy as np
import pandas as pd

# Data Loading (MNIST)

def load_mnist():
    try:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata("MNIST original")
        X, y = mnist["data"], mnist["target"]
    except ImportError:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X, y = mnist.data, mnist.target

    X = X.astype(np.float32) / 255.0
    y = y.astype(int)

    return X, y


# Utility Functions

def one_hot(y, num_classes):
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float32)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(probs, targets):
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.mean(np.sum(targets * np.log(probs), axis=1))


# MLP Model (from scratch)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42, weight_scale=0.01):
        rng = np.random.RandomState(seed)

        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * weight_scale
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)

        self.W2 = rng.randn(hidden_dim, output_dim).astype(np.float32) * weight_scale
        self.b2 = np.zeros((1, output_dim), dtype=np.float32)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = softmax(z2)
        return probs, (X, z1, a1, probs)

    def backward(self, cache, targets):
        X, z1, a1, probs = cache
        N = X.shape[0]

        dZ2 = (probs - targets) / N
        dW2 = a1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        da1 = dZ2 @ self.W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads, lr, weight_decay=0.0):
        self.W1 -= lr * (grads["W1"] + weight_decay * self.W1)
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * (grads["W2"] + weight_decay * self.W2)
        self.b2 -= lr * grads["b2"]

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)


# Training Loop

def train(model, X, y, epochs=15, lr=0.05, batch_size=32, weight_decay=1e-3):
    y_onehot = one_hot(y, 10)
    N = X.shape[0]

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(N)
        total_loss = 0.0
        start = time.time()

        for i in range(0, N, batch_size):
            batch_idx = idx[i:i + batch_size]
            Xb = X[batch_idx]
            yb = y_onehot[batch_idx]

            probs, cache = model.forward(Xb)
            loss = cross_entropy_loss(probs, yb)
            grads = model.backward(cache, yb)
            model.step(grads, lr, weight_decay)

            total_loss += loss * Xb.shape[0]

        avg_loss = total_loss / N
        train_acc = np.mean(model.predict(X[:5000]) == y[:5000])

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc (subset): {train_acc:.4f} | "
            f"Time: {time.time() - start:.2f}s"
        )


# Main

def main():
    X, y = load_mnist()

    # Train / test split (portfolio-friendly)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Mean centering
    mean = X_train.mean(axis=0, keepdims=True)
    X_train -= mean
    X_test -= mean

    model = MLP(
        input_dim=784,
        hidden_dim=512,
        output_dim=10
    )

    train(model, X_train, y_train)

    test_preds = model.predict(X_test)
    test_acc = np.mean(test_preds == y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    pd.DataFrame({
        "index": np.arange(len(test_preds)),
        "predicted_label": test_preds
    }).to_csv("predictions.csv", index=False)

    print("Saved predictions.csv")


if __name__ == "__main__":
    main()
