import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data / 16.0
y = digits.target

num_classes = 10
y_onehot = np.eye(num_classes)[y]

X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
    X, y_onehot, y, test_size=0.30, stratify=y, random_state=42
)

print(f"[1] Dataset prepared: Train={X_train.shape}, Test={X_test.shape}")


class TwoLayerNet:
    def __init__(self, input_dim=64, hidden_width=16, output_dim=10, seed=42, weight_std=0.1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, weight_std, size=(input_dim, hidden_width))
        self.b1 = np.zeros(hidden_width)

        self.W2 = rng.normal(0.0, weight_std, size=(hidden_width, hidden_width))
        self.b2 = np.zeros(hidden_width)

        self.W3 = rng.normal(0.0, weight_std, size=(hidden_width, output_dim))
        self.b3 = np.zeros(output_dim)

        self.cache = {}
        self.grads = {}

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def softmax(z):
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.W3 + self.b3
        probs = self.softmax(z3)

        self.cache = {
            'X': X, 'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2, 'z3': z3, 'probs': probs
        }
        return probs

    @staticmethod
    def cross_entropy_loss(probs, y_true):
        batch_size = y_true.shape[0]
        clipped_probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
        loss = -np.sum(y_true * np.log(clipped_probs)) / batch_size
        return loss

    def backward(self, y_true):
        X = self.cache['X']
        z1, a1 = self.cache['z1'], self.cache['a1']
        z2, a2 = self.cache['z2'], self.cache['a2']
        probs = self.cache['probs']
        batch_size = X.shape[0]

        dz3 = (probs - y_true) / batch_size
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        self.grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3
        }
        return self.grads

    def step(self, lr):
        self.W1 -= lr * self.grads['W1']
        self.b1 -= lr * self.grads['b1']
        self.W2 -= lr * self.grads['W2']
        self.b2 -= lr * self.grads['b2']
        self.W3 -= lr * self.grads['W3']
        self.b3 -= lr * self.grads['b3']

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


net_check = TwoLayerNet(hidden_width=16, seed=42, weight_std=0.01)
probs_sample = net_check.forward(X_train[:32])

assert probs_sample.shape == (32, 10), "Output shape mismatch! Expected (32, 10)."
assert np.allclose(np.sum(probs_sample, axis=1), 1.0, atol=1e-6), "Row probabilities do not sum to 1!"

initial_loss = net_check.cross_entropy_loss(probs_sample, y_train[:32])
print("[3] Forward pass check passed: shape is (32, 10) and rows sum to 1 within 1e-6.")
print(f"[4] Untrained network initial loss: {initial_loss:.4f} (expected ~ln(10) = 2.3026)")


def gradient_check(X_batch, y_batch, num_checks=10, eps=1e-5):
    net = TwoLayerNet(hidden_width=16, seed=42, weight_std=0.1)
    net.forward(X_batch)
    net.backward(y_batch)

    rng = np.random.default_rng(10)
    weight_names = ['W1', 'W2', 'W3']
    passed = 0

    print("\n[5] Running numerical gradient check on 10 random weights:")
    for _ in range(num_checks):
        w_name = rng.choice(weight_names)
        W_matrix = getattr(net, w_name)
        idx = (int(rng.integers(0, W_matrix.shape[0])), int(rng.integers(0, W_matrix.shape[1])))

        orig_val = W_matrix[idx]
        analytic_grad = net.grads[w_name][idx]

        W_matrix[idx] = orig_val + eps
        loss_plus = net.cross_entropy_loss(net.forward(X_batch), y_batch)

        W_matrix[idx] = orig_val - eps
        loss_minus = net.cross_entropy_loss(net.forward(X_batch), y_batch)

        W_matrix[idx] = orig_val

        num_grad = (loss_plus - loss_minus) / (2 * eps)
        rel_error = abs(analytic_grad - num_grad) / max(1e-12, abs(analytic_grad) + abs(num_grad))

        is_ok = rel_error < 1e-6
        if is_ok:
            passed += 1
        print(
            f"  Weight {w_name}{idx}: Analytic={analytic_grad:.6e}, Numerical={num_grad:.6e}, RelErr={rel_error:.2e} [{'PASS' if is_ok else 'FAIL'}]")

    print(f"Gradient check result: {passed}/{num_checks} weights passed (< 1e-6 relative error).\n")


gradient_check(X_train[:16], y_train[:16])

os.makedirs("../results", exist_ok=True)
widths = [16, 64]
learning_rates = [0.001, 0.01, 0.1]
epochs = 50
batch_size = 32

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

print("[6] Training models (50 epochs, mini-batch size 32):")
for idx_w, W in enumerate(widths):
    ax = axes[idx_w]
    for lr in learning_rates:
        net = TwoLayerNet(hidden_width=W, seed=42, weight_std=0.1)
        rng = np.random.default_rng(42)
        epoch_losses = []

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(X_train.shape[0])
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            batch_losses = []
            for start in range(0, X_train.shape[0], batch_size):
                X_b = X_shuffled[start:start + batch_size]
                y_b = y_shuffled[start:start + batch_size]

                probs = net.forward(X_b)
                loss = net.cross_entropy_loss(probs, y_b)
                batch_losses.append(loss)

                net.backward(y_b)
                net.step(lr)

            mean_loss = np.mean(batch_losses)
            epoch_losses.append(mean_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  [W={W:2d} | lr={lr:<5}] Epoch {epoch:2d}/{epochs} - Loss: {mean_loss:.4f}")

        test_preds = net.predict(X_test)
        test_acc = np.mean(test_preds == y_test_raw) * 100
        print(
            f"  --> FINAL [W={W:2d}, lr={lr:<5}] Train Loss: {epoch_losses[-1]:.4f} | Test Accuracy: {test_acc:.2f}%\n")

        ax.plot(range(1, epochs + 1), epoch_losses, label=f"lr={lr} (Acc: {test_acc:.1f}%)")

    ax.set_title(f"Hidden Width W = {W}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("results/loss_curves.png", dpi=150)
print("Saved loss curves plot to: results/loss_curves.png\n")

print("[7] Timing forward and backward passes (Median of 100 runs, batch_size=32):")
for W in widths:
    net = TwoLayerNet(hidden_width=W, seed=42, weight_std=0.1)
    X_b = X_train[:32]
    y_b = y_train[:32]

    fwd_times = []
    bwd_times = []

    for _ in range(100):
        t0 = time.perf_counter()
        net.forward(X_b)
        t1 = time.perf_counter()
        net.backward(y_b)
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1e6)
        bwd_times.append((t2 - t1) * 1e6)

    med_fwd = np.median(fwd_times)
    med_bwd = np.median(bwd_times)
    print(
        f"  Width W={W:2d} | Forward median: {med_fwd:7.2f} µs | Backward median: {med_bwd:7.2f} µs | Ratio (Bwd/Fwd): {med_bwd / med_fwd:.2f}x")