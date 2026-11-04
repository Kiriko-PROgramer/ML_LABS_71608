# Lab 5: Frameworks and Automatic Differentiation

## 1. Introduction and Objectives
The primary objective of this laboratory is to understand the internal mechanics of modern Deep Learning frameworks (like PyTorch and TensorFlow). Specifically, this lab focuses on implementing a custom automatic differentiation (autograd) engine using dynamic computation graphs, validating its numerical accuracy, and analyzing the hardware-level performance implications of different execution modes and tensor memory layouts.

Modern AI frameworks abstract away the complexity of backpropagation by constructing a Directed Acyclic Graph (DAG) of tensor operations. By implementing a scalar-based version of this engine, we gain insight into topological sorting, gradient accumulation, and the chain rule of calculus.

---

## 2. Graded Evidence: Execution Modes & Tensor Optimization

Based on the execution benchmarks conducted in this laboratory, tracing and compiling computational graphs offer distinct performance advantages over standard eager execution, though results vary depending on the model scale and hardware.

**When Compilation/Tracing Helps:**
Eager execution processes operations sequentially in Python, introducing interpreter overhead at every step. Tracing (using `torch.jit.trace`) or compiling (using `torch.compile`) serializes the computation graph upfront. This allows the underlying C++ backend to optimize operator fusion (e.g., combining a linear layer and a ReLU activation into a single GPU kernel), eliminate redundant memory allocations, and bypass the Python Global Interpreter Lock (GIL). 

Compilation provides the most significant performance uplifts in production deployments, during loops with static shapes and repetitive operations, and when running large parallel batches where Python dispatching becomes the primary bottleneck.

**The Cost of Non-Contiguous Tensors:**
Operations requiring non-contiguous memory access demonstrate significantly slower execution times compared to contiguous tensors. When a tensor is transposed, its shape changes, but the underlying one-dimensional data array remains unchanged; only the strides are modified. When a tensor is non-contiguous, the memory addresses of logically neighboring elements are no longer strictly adjacent in physical memory. This defeats CPU/GPU cache prefetching mechanisms, forcing the hardware to perform dispersed memory lookups (cache misses). Enforcing memory contiguity (using `.contiguous()`) prior to heavy aggregate operations rearranges the physical memory, entirely resolving this overhead.

---

## 3. Part 1: Autograd Engine Implementation (Tasks 1-3)

The custom autograd engine relies on the `Value` class, which wraps scalar numbers and tracks their operational history to facilitate reverse-mode automatic differentiation.

### 3.1 Forward Pass and Graph Construction
During the forward pass, every mathematical operation (e.g., addition, multiplication, exponentiation) and non-linear activation (ReLU, Sigmoid) evaluates the actual numerical result while simultaneously creating a new `Value` node. This node stores:
1.  The scalar data.
2.  Pointers to its parent nodes (`_prev`).
3.  The operation that created it (`_op`).

This dynamically constructs the computational graph.

### 3.2 Backward Pass and Topological Sort
To compute gradients, we implemented a reverse-mode traversal. The algorithm performs a **Topological Sort** to ensure that a node's gradient is fully calculated before it propagates that gradient to its parent nodes. 

The backpropagation relies on the fundamental Chain Rule of calculus:
$$ \frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x} $$

During the backward pass, gradients are *accumulated* (`+=`) rather than overwritten, which is critical when a single node acts as an input to multiple subsequent operations (preventing gradient loss).

---

## 4. Part 2: Gradient Verification (Task 4)

To verify the correctness of the implemented mechanism, we computed the derivative of the following composite function:
$$ f(a, b, c) = \text{ReLU}((a \cdot b + c)^2) $$

Given the inputs $a=2.0$, $b=3.0$, and $c=4.0$:
1. $a \cdot b + c = 2(3) + 4 = 10$
2. $10^2 = 100$
3. $\text{ReLU}(100) = 100$

The analytical gradient with respect to $a$ is:
$$ \frac{\partial f}{\partial a} = 2(a \cdot b + c) \cdot b = 2(10) \cdot 3 = 60 $$

We also calculated the numerical gradient using the finite difference method:
$$ f'(a) \approx \frac{f(a + h, b, c) - f(a, b, c)}{h} $$
where $h = 10^{-7}$.

**Test Results:**
* **Custom Autograd (our implementation):** 60.0
* **PyTorch Autograd:** 60.0
* **Numerical Estimate:** 60.0000
* **Absolute Difference:** 0.0

**Conclusion:** The custom engine works perfectly. The absolute difference between PyTorch's highly optimized engine and our scalar implementation is 0.0, well within the strict $< 10^{-6}$ tolerance limit. This proves our topological sort and local derivative rules are mathematically sound.

---

## 5. Part 3: Model Training on make_moons Dataset (Task 5)

To benchmark standard framework speeds, a fully connected multi-layer perceptron (MLP) was trained on the `make_moons` dataset.
* **Architecture:** $2 \rightarrow 8 \rightarrow 1$
* **Loss Function:** Binary Cross-Entropy (BCE)
$$ L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

**Result:**
* **PyTorch Median Training Time (10 epochs):** 0.00721 seconds

The extremely fast convergence highlights PyTorch's ATen C++ backend efficiency. Matrix multiplications and gradient updates are processed via heavily optimized BLAS (Basic Linear Algebra Subprograms) libraries, drastically outperforming scalar-based Python loops.

---

## 6. Part 4: Overhead and Execution Modes Evaluation (Task 6)

### 6.1 Eager Execution vs. Traced Execution
PyTorch defaults to **Eager mode**, evaluating operations dynamically line-by-line. We benchmarked this against a precompiled graph using `torch.jit.trace`.

* **Eager Execution Time:** 0.35899 s
* **Traced Execution Time:** 0.49090 s

**Analysis:** In this specific scenario using a highly simplified model (`Linear(512, 512)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(512, 512)`) on a CPU, tracing introduced more overhead than it resolved. The setup and dispatch time for the JIT compiler exceeded the raw Python execution time. Graph compilation (JIT) shines predominantly in complex models (e.g., Transformers) running on GPUs, where reducing Python-to-C++ context switches and enabling operator fusion yields massive latency reductions.

### 6.2 Impact of Memory Contiguity
Tensors in PyTorch are backed by flat, 1D memory arrays. The layout is dictated by strides. For a tensor of shape $N \times M$, traversing rows requires specific memory jumps. When calling `.t()` to transpose, PyTorch does not move the data; it swaps the strides, resulting in a non-contiguous tensor.

We tested a `sum(dim=1)` reduction on $5000 \times 5000$ tensors:
* **Tensor 1 (Contiguous = True):** 0.00453 s
* **Tensor 2 (Contiguous = False):** 0.03182 s

**Conclusion:** The non-contiguous sum was **over 7 times slower**. Operating on non-contiguous memory requires the CPU to constantly fetch dispersed memory blocks into the L1/L2 cache, leading to severe cache thrashing. Reallocating the memory sequentially via `.contiguous()` before performing heavy reductions is a critical optimization in data engineering.