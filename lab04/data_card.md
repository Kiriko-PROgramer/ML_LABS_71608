# Data Card: Adult Income Dataset (UCI)

## 1. Dataset Overview & Provenance
- **Original Source:** 1994 United States Census Bureau database.
- **Donors:** Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics).
- **Extraction Conditions:** The original extraction filtered out children (age < 16), people not in the workforce (hours-per-week < 1), and those with zero demographic weight (fnlwgt < 1). 
- **Hosting:** UCI Machine Learning Repository and OpenML (version 2).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

## 2. Dataset Composition & Size
- **Original Dataset:** 48,842 instances, 14 predictive features, 1 target variable (`income`).
- **Target Variable:** Binary classification task predicting whether an individual earns `>50K` or `<=50K` annually.
- **Training Set (Post-Split & Deduplication):** ~34,189 instances.
- **Test Set (Unaltered Holdout):** 14,653 instances.
- **Missing Values:** Represented as `?` in the original dataset (handled via imputation in our pipeline).

## 3. Feature Information
- **Numeric Features:** `age`, `fnlwgt` (final weight), `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
- **Categorical Features:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`.
- **Engineered Features:** 
  - `capital_net` (Capital Gain minus Capital Loss).
  - `hours_per_age` (Hours worked per week normalized by working-age years).

## 4. Known Biases & Ethical Considerations
- **Temporal & Economic Bias:** The income threshold of $50,000 is based on the 1994 economic climate. Due to decades of inflation, this dataset does not reflect modern purchasing power or contemporary poverty/wealth lines.
- **Historical Representation Bias:** The data captures structural biases present in the 1990s US labor market, including significant wage disparities across gender and racial groups, as well as occupational segregation. Models trained on this data risk learning and perpetuating these historical inequities.
- **Geographic Bias:** The dataset is strictly representative of the United States in 1994 and will not generalize to other countries or regions.

## 5. Privacy & Security Concerns
- **PII Removal:** Direct Personally Identifiable Information (PII) such as names, exact addresses, and Social Security Numbers were scrubbed by the Census Bureau prior to release.
- **De-anonymization Risk (Mosaic Effect):** While anonymized, individuals with rare combinations of attributes (e.g., specific high capital gains, rare occupations, distinct ages, and minority native countries) could theoretically be re-identified if cross-referenced with external historical public records. 

## 6. Pipeline & Processing Summary
- **Imputation:** Numeric missing values filled with the training set median; Categorical missing values filled with the training set mode.
- **Outlier Handling:** Numeric features clipped to the 1st and 99th percentiles (bounds calculated strictly on training data to prevent leakage).
- **Leakage Prevention:** Deduplication applied *only* to the training set; Test set kept perfectly isolated.
- **Reproducibility:** Transformation logic saved as `preprocessor.joblib`.
- **Processed Data SHA-256 Hash:** `d17a63510e8868c1fe7bf3ccfdd331266f6dd8a2a3138542671398fca2d2ded4`