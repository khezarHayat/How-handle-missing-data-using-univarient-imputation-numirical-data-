# Handling Missing Data Using Univariate Imputation — Numerical Data

This repository contains my practical work on **handling missing numerical data using Univariate Imputation techniques** in Machine Learning.

Missing values are common in real-world datasets. Numerical features such as age, salary, experience, price, and income can contain missing values for many reasons. Before using these features to train a Machine Learning model, the missing values need to be handled appropriately.

In this practice, I focused specifically on **numerical data** and explored different univariate imputation techniques.

## What is Univariate Imputation?

Univariate imputation means filling missing values in a feature by using information from **that same feature**.

For example:

```text
Age
20
25
NaN
30
35
```

The missing `Age` value can be estimated using the available values in the `Age` column.

```text
Age
20
25
27.5
30
35
```

The value used for imputation depends on the selected technique.

---

# Techniques Practiced

## 1. Mean Imputation

Mean imputation replaces missing values with the **average value** of the numerical feature.

For example:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

Mean imputation is generally more appropriate when the feature distribution is reasonably symmetric and does not contain strong outliers.

---

## 2. Median Imputation

Median imputation replaces missing values with the **middle value** of the feature after sorting the available observations.

```python
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

Median imputation is useful when the numerical feature is skewed or contains outliers because the median is less sensitive to extreme values.

---

## 3. Arbitrary Value Imputation

In arbitrary value imputation, missing values are replaced with a specific value chosen by the practitioner.

For example:

```python
imputer = SimpleImputer(
    strategy="constant",
    fill_value=-999
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

The chosen value should make sense for the particular dataset and modeling problem.

---

## 4. Random Sample Imputation

Random sample imputation replaces each missing value with a randomly selected value from the **observed values of the same feature**.

For example:

```text
Age
20
25
NaN
30
35
```

The missing value could be replaced with one of the observed values, such as `25` or `30`.

This technique can help preserve the distribution of the original feature because actual observed values are used instead of a single fixed statistic.

---

# Comparing the Techniques

| Technique | Replacement Value | Useful When |
|---|---|---|
| Mean | Average | Data is reasonably symmetric |
| Median | Middle value | Data is skewed or has outliers |
| Arbitrary Value | User-defined value | Missingness itself may be informative |
| Random Sample | Existing random value | Preserving the feature distribution is important |

---

# Important Machine Learning Practice

The imputation value should be calculated **only from the training data**.

Correct approach:

```python
imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
```

We should not calculate the mean or median using the complete dataset before splitting because this can cause **data leakage**.

---

# Workflow

The workflow practiced in this repository is:

```text
Raw Dataset
     ↓
Identify Missing Numerical Values
     ↓
Split Data
     ↓
Select Imputation Technique
     ↓
Fit Imputer on Training Data
     ↓
Transform Training Data
     ↓
Transform Test Data
     ↓
Train ML Model
     ↓
Evaluate Model
```

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

# Learning Goal

The goal of this repository is to develop a practical understanding of **Univariate Imputation for Numerical Data** and learn how different techniques can be used to handle missing numerical values before training Machine Learning models.

This practice is part of my ongoing **Machine Learning and AI learning journey**.
