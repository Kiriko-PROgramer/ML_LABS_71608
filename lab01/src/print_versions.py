import sys
from importlib.metadata import version, PackageNotFoundError

def print_versions():
    print(f"Python: {sys.version}")
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "psutil"]
    for pkg in packages:
        try:
            print(f"{pkg}: {version(pkg)}")
        except PackageNotFoundError:
            print(f"{pkg}: NOT FOUND")

if __name__ == "__main__":
    with open("../results/versions.txt", "w") as f:
        sys.stdout = f
        print_versions()
        sys.stdout = sys.__stdout__