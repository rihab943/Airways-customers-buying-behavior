# British Airways Customer Booking Behavior

## Project Overview

This project explores customer booking behavior for British Airways and builds machine-learning classification models to predict whether a customer completes a booking. The workflow includes exploratory data analysis (EDA), preprocessing, feature encoding, model training, evaluation, and feature-importance analysis.

## Dataset

The project uses a customer-booking dataset containing **50,000 rows** and **14 original columns**.

The target variable is `booking_complete`:
- `1`: booking completed
- `0`: booking not completed

The target is imbalanced in the dataset:
- `0`: 42,522 records (85.044%)
- `1`: 7,478 records (14.956%)

The original columns include passenger count, sales channel, trip type, purchase lead time, length of stay, flight hour and day, route, booking origin, optional-service preferences, flight duration, and booking completion.

> Update the dataset paths in the notebooks/scripts to match your local environment. The source notebooks currently use absolute paths.

## Workflow

### 1. Exploratory Data Analysis and Preprocessing

The EDA workflow:
- Loads the source CSV using `pandas`.
- Inspects columns, data types, missing values, and possible outliers or negative values.
- Examines class imbalance.
- Compares numerical-feature distributions by booking outcome.
- Compares completion rates across low-cardinality categories such as sales channel, trip type, and flight day.
- Examines numerical correlations.
- Investigates booking completion rates by booking origin and route.

### 2. Feature Encoding

The preprocessing workflow encodes categorical features:
- **One-hot encoding** for `sales_channel`, `trip_type`, and `flight_day`.
- **Target encoding** for `route` and `booking_origin`.

The resulting encoded dataset is saved as `encoded_customer_booking.csv`.


### 3. Model Training and Evaluation

The model notebook trains and evaluates:
- **Logistic Regression**, with `class_weight="balanced"`.
- **Random Forest Classifier**, with 100 trees and `class_weight="balanced"`.

The data is split using `train_test_split` with `test_size=0.2` and `random_state=42`.

The notebook evaluates predictions using accuracy, precision, recall, and F1-score. It also examines Random Forest probability thresholds and calculates average precision (PR-AUC / average precision), confusion-matrix-related metrics, and a classification report.

### 4. Feature Importance and Cross-Validation

Permutation importance is calculated on the test set using F1-score as the scoring metric. In the recorded results, the strongest features by permutation importance include:
1. `route`
2. `booking_origin`
3. `flight_duration`
4. `length_of_stay`
5. `purchase_lead`

The Random Forest is also evaluated with 7-fold stratified cross-validation using F1-score. The notebook reports a mean cross-validation F1-score of `0.1544`.

## Recorded Results

### Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 0.7225 |
| Precision | 0.3106 |
| Recall | 0.7176 |
| F1-score | 0.4336 |

The notebook reports a convergence warning for Logistic Regression and suggests increasing `max_iter` and/or scaling features.

### Random Forest at the Default Threshold

| Metric | Score |
|---|---:|
| Accuracy | 0.8570 |
| Precision | 0.5880 |
| Recall | 0.1128 |
| F1-score | 0.1893 |

At the default 0.5 probability threshold, the Random Forest has relatively high precision but low recall for completed bookings.

### Random Forest at a 0.22 Threshold

| Metric | Score |
|---|---:|
| Accuracy | 0.7700 |
| Precision for class 1 | 0.35 |
| Recall for class 1 | 0.63 |
| F1-score for class 1 | 0.45 |
| Average precision (reported as PR-AUC) | 0.4015 |

The notebook selects a probability threshold of `0.22` to improve detection of completed bookings. This trades some precision and accuracy for higher recall.

### Cross-Validation

The recorded mean 7-fold cross-validation F1-score for the Random Forest is **0.1544**. This differs notably from the threshold-adjusted test F1-score. Check that cross-validation uses the same decision threshold and that preprocessing is performed inside each fold before comparing the scores.

## Repository Structure

The expected project layout is approximately:

```text
.
├── data/
│   ├── customer_booking.csv
│   └── encoded_customer_booking.csv
├── notebooks/
│   ├── 01_Exploring_data_eda_preprocessing_feature_eng.ipynb
│   └── 02_model_and_evaluation.ipynb
├── outputs/
│   └── Feature_importance.png
├── src/
│   └── Exploring_data_functions.py
├── requirement.md
└── README.md
```


## Requirements

See requirement.txt

## How to Run

1. Clone or open the project repository.
2. Install the dependencies.
3. Place `customer_booking.csv` in the `data/` directory.
4. Update any absolute file paths so they point to the local project directory.
5. Run the EDA/preprocessing notebook first to generate `encoded_customer_booking.csv`.
6. Run the model/evaluation notebook.

## Notes on the Results

- The target variable is imbalanced: about 15% of records represent completed bookings.
- The notebooks compare model metrics at different Random Forest probability thresholds. At a threshold of `0.22`, recall for completed bookings is higher than at the default threshold.
- Logistic Regression produced a convergence warning during training.
- The reported 7-fold cross-validation mean F1-score for the Random Forest is `0.1544`.


