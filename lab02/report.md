# Laboratory Report: Lab 2 – A Neural Network from Scratch
**Course:** Machine Learning (CII5SE039CI)  
**Student ID:** 71608  

---

## 1. Dataset & Execution Overview
The script successfully loaded and preprocessed the `load_digits` dataset, performed stratified splitting ($70/30$), initialised the network, verified the forward pass and initial loss, executed numerical gradient checking, trained models across multiple learning rates and widths, and measured execution timings.

### Console Output Log (`OUTPUT.txt`)
```text
C:\Users\Kiriya\Desktop\lessons\lab02\.venv\Scripts\python.exe C:\Users\Kiriya\Desktop\lessons\lab02\main.py 
[1] Dataset prepared: Train=(1257, 64), Test=(540, 64)
[3] Forward pass check passed: shape is (32, 10) and rows sum to 1 within 1e-6.
[4] Untrained network initial loss: 2.3026 (expected ~ln(10) = 2.3026)

[5] Running numerical gradient check on 10 random weights:
  Weight W3(15, 2): Analytic=6.127157e-03, Numerical=6.127157e-03, RelErr=2.51e-09 [PASS]
  Weight W1(50, 13): Analytic=0.000000e+00, Numerical=0.000000e+00, RelErr=0.00e+00 [PASS]
  Weight W2(2, 13): Analytic=3.645803e-03, Numerical=3.645803e-03, RelErr=4.90e-10 [PASS]
  Weight W2(2, 2): Analytic=0.000000e+00, Numerical=0.000000e+00, RelErr=0.00e+00 [PASS]
  Weight W2(11, 6): Analytic=0.000000e+00, Numerical=0.000000e+00, RelErr=0.00e+00 [PASS]
  Weight W3(0, 4): Analytic=1.809229e-04, Numerical=1.809229e-04, RelErr=3.46e-08 [PASS]
  Weight W2(15, 3): Analytic=2.244256e-03, Numerical=2.244256e-03, RelErr=7.09e-10 [PASS]
  Weight W3(1, 3): Analytic=0.000000e+00, Numerical=0.000000e+00, RelErr=0.00e+00 [PASS]
  Weight W3(9, 9): Analytic=4.535897e-03, Numerical=4.535897e-03, RelErr=9.28e-10 [PASS]
  Weight W3(14, 8): Analytic=2.637842e-03, Numerical=2.637842e-03, RelErr=2.15e-09 [PASS]
Gradient check result: 10/10 weights passed (< 1e-6 relative error).

[6] Training models (50 epochs, mini-batch size 32):
  [W=16 | lr=0.001] Epoch 50/50 - Loss: 2.2977 --> Test Accuracy: 17.78%
  [W=16 | lr=0.01 ] Epoch 50/50 - Loss: 1.4363 --> Test Accuracy: 65.56%
  [W=16 | lr=0.1  ] Epoch 50/50 - Loss: 0.0383 --> Test Accuracy: 96.48%

  [W=64 | lr=0.001] Epoch 50/50 - Loss: 2.2299 --> Test Accuracy: 35.74%
  [W=64 | lr=0.01 ] Epoch 50/50 - Loss: 0.3073 --> Test Accuracy: 91.30%
  [W=64 | lr=0.1  ] Epoch 50/50 - Loss: 0.0151 --> Test Accuracy: 96.85%

[7] Timing forward and backward passes (Median of 100 runs, batch_size=32):
  Width W=16 | Forward median:   74.30 µs | Backward median:   83.50 µs | Ratio (Bwd/Fwd): 1.12x
  Width W=64 | Forward median:  107.85 µs | Backward median:  141.80 µs | Ratio (Bwd/Fwd): 1.31x