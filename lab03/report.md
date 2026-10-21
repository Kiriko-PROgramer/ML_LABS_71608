# Lab 3: Architectures and their Costs

## 1. Experimental Setup and Results
This report compares three fundamental neural network architectures (MLP, CNN, and Attention/Transformer) on the Fashion-MNIST dataset. All models were trained for 5 epochs using the Adam optimizer (learning rate = 1e-3) with a batch size of 128.

### Empirical Test Accuracy (After 5 Epochs)
* **CNN:** 89.11% *(Best)*
* **MLP:** 87.92%
* **Attention:** 83.98%

---

## 2. Cost Calculations: Analytical vs. Profiled FLOPs

### Hand-Calculated FLOPs (Single Layer)
*   **MLP (First Linear Layer: 784 $\rightarrow$ 256):** 
    $$FLOPs = 2 \times N_{in} \times N_{out} = 2 \times 784 \times 256 = 401,408$$
*   **CNN (First Conv2d: 1 $\rightarrow$ 16, $3\times3$ kernel, output $28\times28$):** 
    $$FLOPs = 2 \times K^2 \times C_{in} \times C_{out} \times H_{out} \times W_{out} = 2 \times 9 \times 1 \times 16 \times 28 \times 28 = 225,792$$
*   **Attention (Q, K, V projections for seq=16, d_model=64):** 
    $$FLOPs = 3 \times (2 \times seq \times d\_model^2) = 3 \times 2 \times 16 \times 64 \times 64 = 393,216$$

### Gap Explanation (Manual vs. `thop` Profiler)
While hand calculations isolate the primary matrix multiplications of a single layer, the `thop.profile` tool measures the **entire forward pass**. The gap exists because the profiler includes:
1. Subsequent layers (e.g., Transformer FeedForward networks, final classification heads).
2. Activation functions (ReLU), Pooling operations (MaxPool), and Normalization layers.
3. Bias additions and flattening operations.

---

## 3. Architecture Trade-offs and Access Patterns

Based on the generated `architecture_tradeoffs.png` plots, we can observe distinct trade-offs between accuracy, parameter count, and latency.

### Accuracy vs. Parameters
*   **CNN:** Achieved the highest accuracy (89.11%) with the **lowest** parameter count. This is due to *weight sharing* and *local receptive fields*. The CNN efficiently extracts spatial hierarchies without needing a parameter for every pixel combination.
*   **MLP:** Required the **highest** number of parameters because every input pixel is densely connected to every neuron in the first layer, yet it achieved lower accuracy than the CNN due to a lack of spatial inductive bias.
*   **Attention:** Has a moderate parameter count but yielded the lowest accuracy (83.98%). Transformers generally lack the structural priors of CNNs and typically require much larger datasets (or significantly more epochs/data augmentation) to learn spatial relationships from scratch.

### Accuracy vs. Latency (Access Patterns)
*   **MLP (Lowest Latency):** Matrix multiplications in MLPs have highly regular, contiguous memory access patterns. Modern hardware (GPUs/CPUs) executes these dense GEMM (General Matrix Multiply) operations extremely fast, resulting in low latency.
*   **CNN (Moderate Latency, Highest Accuracy):** CNNs have high FLOPs due to the sliding window mechanism, but they utilize highly optimized spatial access patterns (e.g., `im2col` transformations). They offer the best balance.
*   **Attention (Highest Latency):** The self-attention mechanism requires comparing every patch to every other patch ($O(N^2)$ complexity). This dynamic routing creates irregular memory access patterns and demands high memory bandwidth (memory-bound operations). Consequently, despite having fewer parameters than the MLP, the Attention model exhibits the highest latency.

## 4. Conclusion
For the Fashion-MNIST dataset, the **CNN is definitively the most cost-effective architecture**. It optimally leverages spatial locality and parameter sharing to achieve the highest accuracy, while maintaining a highly efficient computational profile and reasonable latency.