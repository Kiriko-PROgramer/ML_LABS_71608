import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import hashlib
import warnings

warnings.filterwarnings('ignore')


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for i in range(X_df.shape[1]):
            self.bounds_[i] = (X_df[i].quantile(0.01), X_df[i].quantile(0.99))
        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        for i in range(X_df.shape[1]):
            X_df[i] = X_df[i].clip(lower=self.bounds_[i][0], upper=self.bounds_[i][1])
        return X_df.values


def main():
    print("1. Ingesting data...")
    adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
    df = adult.frame
    df.replace('?', np.nan, inplace=True)

    df.rename(columns={'class': 'income'}, inplace=True)

    print("\n2. --- Validation Report ---")
    range_checks = {'age': (17, 90), 'hours-per-week': (1, 99),
                    'capital-gain': (0, 99999), 'capital-loss': (0, 4356)}
    for col, (min_v, max_v) in range_checks.items():
        violations = df[(df[col] < min_v) | (df[col] > max_v)]
        print(f"{col}: {len(violations)} rows out of bounds ({min_v}-{max_v}).")

    dup_conflict = df.groupby(list(df.columns[:-1]), dropna=False, observed=True)['income'].nunique()
    print(f"Label Audit: Found {(dup_conflict > 1).sum()} duplicate feature rows with conflicting labels.")

    print("\n3. Cleaning & Transformation (Leakage-Free)...")
    X = df.drop('income', axis=1)
    y_bin = (df['income'] == '>50K').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, stratify=y_bin, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1).drop_duplicates()
    X_train, y_train = train_df.drop('income', axis=1), train_df['income']
    print(f"Deduplication: Dropped duplicates from train only.")

    print("4. Feature Engineering...")

    def eng_features(df_in):
        df_out = df_in.copy()
        df_out['capital_net'] = df_out['capital-gain'] - df_out['capital-loss']
        df_out['hours_per_age'] = df_out['hours-per-week'] / (df_out['age'] - 16 + 1e-9)
        return df_out

    X_train_eng, X_test_eng = eng_features(X_train), eng_features(X_test)

    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'capital_net',
                'hours_per_age']
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'native-country']

    preprocessor = ColumnTransformer([
        ('num',
         Pipeline([('imp', SimpleImputer(strategy='median')), ('clip', OutlierClipper()), ('scale', StandardScaler())]),
         num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

    pipe_eng = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))])
    pipe_eng.fit(X_train_eng, y_train)

    acc, f1 = accuracy_score(y_test, pipe_eng.predict(X_test_eng)), f1_score(y_test, pipe_eng.predict(X_test_eng))
    print(f"\nModel Performance (With Eng. Features):\nAccuracy: {acc:.4f} | F1: {f1:.4f}")

    joblib.dump(preprocessor, 'preprocessor.joblib')

    print("\n5. Hashing Processed Dataset...")
    processed_test = preprocessor.transform(X_test_eng)
    pd.DataFrame(processed_test).to_csv('processed_test.csv', index=False)

    hasher = hashlib.sha256()
    with open('processed_test.csv', 'rb') as f:
        hasher.update(f.read())
    print(f"Processed Data SHA-256 Hash: {hasher.hexdigest()}")


if __name__ == '__main__':
    main()