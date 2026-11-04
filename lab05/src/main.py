import math
import time
import torch
import numpy as np
import statistics
from sklearn.datasets import make_moons


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        val = 1 / (1 + math.exp(-self.data))
        out = Value(val, (self,), 'sigmoid')

        def _backward():
            self.grad += (val * (1 - val)) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def task4_verify_gradients():
    print("\n--- Task 4: Gradient Verification ---")

    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    out_custom = ((a * b + c) ** 2).relu()
    out_custom.backward()

    a_pt = torch.tensor(2.0, requires_grad=True)
    b_pt = torch.tensor(3.0, requires_grad=True)
    c_pt = torch.tensor(4.0, requires_grad=True)
    out_pt = torch.relu((a_pt * b_pt + c_pt) ** 2)
    out_pt.backward()

    h = 1e-7
    a_h = Value(2.0 + h)
    out_num = ((a_h * b + c) ** 2).relu()
    grad_num = (out_num.data - out_custom.data) / h

    diff = abs(a.grad - a_pt.grad.item())

    print(f"Custom gradient (a):  {a.grad}")
    print(f"PyTorch gradient (a): {a_pt.grad.item()}")
    print(f"Numeric gradient (a): {grad_num:.4f}")
    print(f"Absolute Difference:  {diff}")
    assert diff < 1e-6, "Gradient difference exceeds 1e-6 tolerance!"
    print("STATUS: SUCCESS. Gradients match.")


def task5_train_make_moons():
    print("\n--- Task 5: make_moons Training Speed Comparison ---")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

    def run_pytorch():
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 8),
            torch.nn.Sigmoid(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid()
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.BCELoss()
        X_pt = torch.tensor(X, dtype=torch.float32)
        y_pt = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        start = time.perf_counter()
        for _ in range(10):
            optimizer.zero_grad()
            out = model(X_pt)
            loss = criterion(out, y_pt)
            loss.backward()
            optimizer.step()
        return time.perf_counter() - start

    pt_times = [run_pytorch() for _ in range(5)]
    print(f"PyTorch Median Time (10 epochs): {statistics.median(pt_times):.5f} seconds")


def task6_inspect_and_compare():
    print("\n--- Task 6: Execution Modes & Tensor Overhead ---")
    device = torch.device('cpu')
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512)
    )
    x = torch.randn(1024, 512)

    _ = model(x)
    traced_model = torch.jit.trace(model, x)
    _ = traced_model(x)

    start = time.perf_counter()
    for _ in range(100): _ = model(x)
    eager_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(100): _ = traced_model(x)
    traced_time = time.perf_counter() - start

    print(f"Eager Execution Time:  {eager_time:.5f} s")
    print(f"Traced Execution Time: {traced_time:.5f} s")

    contig_tensor = torch.randn(5000, 5000)
    non_contig_tensor = contig_tensor.t()

    print(f"\nTensor 1 Contiguous? {contig_tensor.is_contiguous()}")
    print(f"Tensor 2 Contiguous? {non_contig_tensor.is_contiguous()}")

    start = time.perf_counter()
    _ = contig_tensor.sum(dim=1)
    t_contig = time.perf_counter() - start

    start = time.perf_counter()
    _ = non_contig_tensor.sum(dim=1)
    t_non_contig = time.perf_counter() - start

    print(f"Contiguous sum time:     {t_contig:.5f} s")
    print(f"Non-contiguous sum time: {t_non_contig:.5f} s")


if __name__ == '__main__':
    task4_verify_gradients()
    task5_train_make_moons()
    task6_inspect_and_compare()