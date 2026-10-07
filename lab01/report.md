### 1. Goal
Create an isolated Python environment, train baseline Logistic Regression and Random Forest models on the Breast Cancer dataset, and measure their system costs (training time, inference latency, model size, peak memory) to evaluate deployment suitability.

### 2. Method
I created an isolated virtual environment with Python 3.11.8 and installed pinned dependencies. The Breast Cancer dataset was split 70/30 (stratified). Two models were trained with `random_state=42`. Training time was measured as the median of 5 runs (after 1 warm-up). Single-sample inference latency was measured as the median of 100 runs. Model size was calculated from the saved joblib file, and peak memory was measured using `psutil`.

### 3. Results
| Model | Accuracy | Train_Time_s | Inf_Time_ms | Size_KB | Peak_Mem_MB |
|---|---|---|---|---|---|
| LogisticRegression | 0.9415 | 0.37407200000006924 | 0.08664999995744438 | 1.03 | 0.047 |
| RandomForest | 0.9357 | 0.21450770000001285 | 3.6181499999656808 | 284.07 | 0.063 |

### 4. Conclusions
1. **Logistic Regression** is extremely lightweight (~1 KB) and uses minimal memory, making it a perfect fit for **TinyML** devices (budget: size ≤ 100 KB, memory ≤ 256 KB).
2. **Random Forest** is larger (~284 KB) but still easily fits within the **Mobile** deployment budget (budget: size ≤ 10 MB, memory 64-256 MB) and **Edge** budget.
3. Both models are highly efficient and far below the limits for **Cloud** deployment, but Logistic Regression surprisingly yielded slightly better accuracy (94.15%) than Random Forest (93.57%) on this dataset, making it the optimal choice overall.