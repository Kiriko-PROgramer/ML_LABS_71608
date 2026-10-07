import os, time, psutil, joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)


def measure_memory():
    return psutil.Process().memory_info().rss / (1024 * 1024)


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    train_times = []
    mem_before = measure_memory()
    for _ in range(5):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - start)
    mem_after = measure_memory()
    median_train = np.median(train_times)
    peak_mem = mem_after - mem_before

    single_sample = X_test[0].reshape(1, -1)
    inf_times = []
    for _ in range(100):
        start = time.perf_counter()
        model.predict(single_sample)
        inf_times.append(time.perf_counter() - start)
    median_inf = np.median(inf_times) * 1000

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_path = f"../results/{name}_model.joblib"
    joblib.dump(model, model_path)
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024

    results.append({
        "Model": name, "Accuracy": round(acc, 4),
        "Train_Time_s": median_train, "Inf_Time_ms": median_inf,
        "Size_KB": size_kb, "Peak_Mem_MB": peak_mem
    })

df = pd.DataFrame(results)
df.to_csv("../results/baseline_accuracy.csv", index=False)
print("Done! Results are saved in the results folder.")
print(df)