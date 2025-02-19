"""
**Documention for Disease Outbreak Prediction Script**

This script is designed for a disease outbreak prediction challenge. It leverages time-series analysis,
environmental data integration, and ensemble machine learning models to forecast disease outbreaks.

**Script Overview:**

The script performs the following key steps:

1. **Configuration**: Defines column lists, location blacklists, and other configuration parameters.
2. **Data Loading**: Loads training data, test data, and environmental datasets (toilets, waste management, water sources).
3. **Data Aggregation**: Aggregates training data by disease, location, year, month, and health facility.
4. **Feature Engineering**: Creates time-based features (date, time index, time since start) and lag features of the target variable.
5. **Environmental Data Merging**: Integrates environmental datasets with the main dataframes based on location and time keys.
6. **Static Feature Generation**: Generates static features (median, mean, max, etc.) from environmental variables for each time step.
7. **Feature Selection**: Selects relevant features based on correlation with the target variable.
8. **Model Training**: Trains LightGBM and LinearSVR models using a custom cross-validation function and different target transformations (original, difference, ratio).
9. **Ensemble Prediction**: Ensembles predictions from different models and target transformations for each disease.
10. **Final Prediction and Submission**: Generates final predictions and creates a submission CSV file.

**1. Configuration and Setup:**

* **Column Lists:**
    * `cat_cols`: Categorical columns (`Category_Health_Facility_UUID`, `Disease`).
    * `key_cols`: Key columns for merging environmental data (`Transformed_Latitude`, `Transformed_Longitude`, `Year`, `Month`).
    * `loc_cols`: Location columns (`Transformed_Latitude`, `Transformed_Longitude`).
    * `train_cols`: Features used for training, including time, location, and categorical columns.
    * `toilets_cols`: List of environmental features from the toilets, waste management and water sources datasets.
    * `disease_cols`: List of diseases.
    * `dis_others`: List of diseases grouped as 'Others'.
    * `waste_cols`: List of waste management feature columns (derived from `toilets_cols`).
    * `water_cols`: List of water sources feature columns (derived from `toilets_cols`).
    * `grp_cols`: Grouping columns for aggregation.
    * `target`: Target variable name ('Total').
    * `corr_features`: Placeholder for correlated features (initially empty, populated later based on correlation analysis).

* **Blacklist and Unique Locations:**
    * `blacklist_locations`: List of location IDs to be blacklisted (specific locations handled differently, e.g., for Diarrhea).
    * `unique_locations`: List of unique location IDs (used for reference, not directly in modeling in this snippet).

**2. Data Loading:**

* Loads four CSV files from the input directory:
    * `Train.csv`: Training data containing disease counts.
    * `Test.csv`: Test data for prediction.
    * `toilets.csv`: Environmental data related to toilets.
    * `waste_management.csv`: Environmental data related to waste management.
    * `water_sources.csv`: Environmental data related to water sources.

* Initializes 'Total' column in the test dataframe to 0 and 'Predicted_Total' to NaN.

**3. Data Aggregation:**

* `train_sum = train.groupby(...)['Total'].sum().reset_index()`: Aggregates the training data by `Disease`, `Location`, `Year`, `Month`, `Category_Health_Facility_UUID`, and location coordinates, summing the 'Total' cases. This creates a summarized training dataset.

**4. Feature Engineering and Preprocessing Function (`feature_engineering_and_preprocessing`)**

* This function enriches the dataframe with time-based features:
    * `'day'`: Sets day to 1 for creating date objects.
    * `'date'`: Creates a date column from 'Year', 'Month', and 'day'.
    * `'tag'`: Initializes a tag column to 1.
    * `'tag_id'`: Creates a cumulative tag ID within each group (defined by disease, location, year, month, facility).
    * `'time_index'`: Converts the 'date' to a numerical timestamp.
    * Sorts dataframe by 'date'.
    * `'start_date'`:  Calculates the first date for each 'Location'.
    * `'diff_date'`: Calculates the difference in days between 'date' and 'start_date' for each location, representing time elapsed since the start of observation for each location.

* Applies this function to `train_sum`, `train`, `test`, and `test_sum` dataframes.

**5. Lag Feature Engineering Function (`create_lag_features`)**

* This function creates lag features for the target variable:
    * Iterates through a range of lag months (default 36).
    * For each lag month `i`, it shifts the `date` column in a temporary dataframe `temp_df` by `i` months.
    * Merges `temp_df` back into the input dataframe `df` based on 'Disease', 'date', 'Location', and 'tag_id', adding lagged target values as new columns (e.g., 'Total_1', 'Total_2', ... 'Total_36').

* Applied to `train`, `test`, `train_sum`, and `test_sum` using appropriate data aggregations (`data_for_lag`, `data_sum_for_lag`).

**6. Merge Environmental Data Function (`merge_environmental_data`)**

* This function merges environmental data from toilets, waste management, and water sources datasets:
    * Creates a 'hosp_id' column by concatenating 'Disease' and 'Location'.
    * Rounds 'Transformed_Latitude' and 'Transformed_Longitude' to the nearest integer for merging consistency.
    * Merges each environmental dataframe (`toilets`, `waste_management`, `water_sources`) with the input dataframe `df` using `key_cols` ('Transformed_Latitude', 'Transformed_Longitude', 'Year', 'Month') and a left merge.
    * Suffixes are added to the merged columns to distinguish the source of environmental data ('_toilets', '_wm', '_water').

* Applied to `train`, `train_sum`, `test`, and `test_sum`.

**7. Static Feature Generation Function (`get_static_features`)**

* This function generates static statistical features from a list of columns (`cols`):
    * Calculates median, mean, max, sum, product, skewness, kurtosis, standard error of the mean, standard deviation, quantiles (0.75, 0.95, 0.99), and count of non-zero values across the specified columns for each row.
    * Appends these static features to the dataframe with suffixes like 'medianT', 'meanT', etc.

**8. Feature Correlation Analysis:**

* Calculates the correlation between the target variable ('Total') and environmental features (`water_cols + waste_cols + toilets_cols`) in the aggregated training data (`train_sum`) for years before 2023.
* Selects features with an absolute correlation greater than 0.05 and stores them in the `corr_features` list.  These features are considered relevant for modeling.

**9. Centralized Cross-Validation Function (`generic_train_cv`)**

* Implements a generic cross-validation framework for model training and prediction.
* Supports KFold and TimeSeriesSplit cross-validation strategies.
* **Inputs:**
    * `X`: Feature dataframe.
    * `y`: Target series.
    * `valid_df`: Validation dataframe.
    * `test_df`: Test dataframe.
    * `model`: Model object (e.g., LGBMRegressor, LinearSVR).
    * `feature_cols`: List of feature columns to use.
    * `cv_type`: Cross-validation type ('kfold' or 'timeseries').
    * `n_splits`: Number of cross-validation splits.
    * `shuffle`: Whether to shuffle data in KFold.
    * `rs`: Random state for reproducibility.
    * `use_scaler`: Boolean to use RobustScaler for feature scaling.
* **Workflow:**
    * Initializes cross-validation strategy based on `cv_type`.
    * Iterates through cross-validation folds:
        * Splits data into training and validation sets.
        * Optionally applies RobustScaler to features and target.
        * Fits the provided `model` on the training data.
        * Predicts on the validation and test sets.
        * Inverse transforms predictions if scaling was used.
        * Appends validation and test predictions to lists.
    * Returns the mean of validation predictions and test predictions across all folds.

**10. Centralized Model Training Function (`generic_model_trainer`)**

* This function orchestrates the training of models for each disease, using different target transformations and model types.
* **Inputs:**
    * `df_train`: Training dataframe.
    * `df_test`: Test dataframe.
    * `disease_name`: Name of the disease being modeled.
    * `model_params`: Dictionary of model parameters for different model types (e.g., LGBMRegressor, LinearSVR).
    * `train_params`: Dictionary of training parameters (e.g., validation year, training year, target transformations, CV parameters).
* **Workflow:**
    * Retrieves training parameters from `train_params`.
    * Iterates through unique dates in the test dataframe (implicitly iterates through months).
    * For each month:
        * Prepares temporary train and test data.
        * Generates static features for the current month's lag columns.
        * Defines validation set for the current month.
        * **Target Transformations and Model Training:**
            * **Original Target (`use_org_target=True`):** Trains LGBMRegressor and LinearSVR on the original 'Total' target, ensembling their predictions.
            * **Difference Target (`use_diff_target=True`):** Trains LGBMRegressor and LinearSVR on the difference between 'Total' and the lagged median target (`medianT`), ensembling predictions and adding back `medianT` to get final predictions.
            * **Ratio Target (`use_ratio_target=True`):** Trains LGBMRegressor and LinearSVR on the ratio of 'Total' to `medianT`, ensembling predictions and multiplying back by `medianT` to get final predictions.
            * **Median Target (`use_median_target=True`):** Uses the 'medianT' feature directly as the prediction.
        * Stores validation predictions in `df_train` and test predictions in `df_test`.
    * **Important Correction:**
        - **Error Fix**: Inside `generic_model_trainer` function, specifically within `if use_diff_target:`, the line `train_data_diff = train_data_diff.reset_index(drop=True)` was added to resolve the `ValueError: cannot reindex on an axis with duplicate labels`. This error occurred during target difference calculation because the index alignment was failing due to potential duplicate labels after filtering. Resetting the index before target calculation resolves this alignment issue.

    * Returns updated `df_train` and `df_test` with predictions.


**11. Model and Training Parameters:**

* Defines parameter dictionaries for LightGBM models (`lgb_params_*`) with variations for different diseases or model types (general, malaria, intestinal worms, diarrhea, blacklisted diarrhea). These parameters control aspects like number of estimators, objective function, regularization, tree depth, and learning rate.
* `model_parameter_sets`: Dictionary mapping disease categories to their respective model parameter configurations (for LGBM and potentially other models).
* `training_parameter_sets`: Dictionary mapping disease categories to their training configurations, including:
    * `year_valid`: Year for validation set.
    * `train_year`: Year cutoff for training data.
    * Flags to enable different target transformations (`use_org_target`, `use_diff_target`, `use_ratio_target`, `use_median_target`).
    * Cross-validation parameters (`cv_shuffle`, `cv_rs`).
    * Other parameters like `use_scaler`, `na_val`, `par_dec`.

**12. Model Training Execution:**

* Iterates through different disease categories and calls `generic_model_trainer` to train models for:
    * `['Dysentery', 'Typhoid', 'Cholera']` (grouped as 'Others').
    * 'Malaria' (original data and aggregated data).
    * 'Intestinal Worms'.
    * 'Diarrhea' (original data and blacklisted diarrhea - aggregated data).
* Prints messages to indicate the start of training for each disease.

**13. Ensemble Predictions:**

* Implements ensemble strategies for different disease groups.
* For each disease group (Others, Malaria, Intestinal Worms, Diarrhea):
    * Defines `ens_cols_*` lists specifying the prediction types to ensemble (e.g., 'org', 'diff', 'ratio').
    * Calculates ensemble predictions by averaging the predictions from different models/target transformations specified in `ens_cols_*`.
    * Stores ensemble predictions in new columns in `test` and `test_sum` dataframes (e.g., 'Predicted_Total_ens', 'Predicted_Total_ml_ens').

**14. Final Prediction and Submission:**

* Initializes 'Predicted_Total' column in `test` dataframe to 0.
* Defines `blacklist` from `blacklist_locations`.
* Sets conditions (`malaria_cond`, `inst_cond`, `dr_cond`, `blacklist_cond`, `other_cond`) to identify rows in the test set belonging to each disease category and blacklist status.
* Assigns final 'Predicted_Total' values based on the ensemble predictions calculated in the previous step, applying different ensemble strategies for different disease conditions.
* Creates a submission dataframe `sub` with 'ID' and 'Predicted_Total' columns.
* Clips 'Predicted_Total' values to be non-negative.
* Saves the submission dataframe to 'test_sub.csv'.
* Prints descriptive statistics of the 'Predicted_Total' column.

**Error Resolution Highlight:**

* **ValueError: cannot reindex on an axis with duplicate labels:** This error was resolved by adding `train_data_diff = train_data_diff.reset_index(drop=True)` in the `generic_model_trainer` function within the `if use_diff_target:` block. This ensures that the index is reset before performing the subtraction operation for creating the difference target, preventing index alignment issues caused by potentially duplicated index labels after data filtering within the function.

**Usage Instructions:**

1. **Install Libraries:** Ensure all necessary libraries are installed (`pandas`, `numpy`, `sklearn`, `lightgbm`, `tqdm`).
2. **Data Setup:** Place the input CSV files (`Train.csv`, `Test.csv`, `toilets.csv`, `waste_management.csv`, `water_sources.csv`) in the same directory as the script or adjust file paths accordingly.
3. **Run the Script:** Execute the Python script.
4. **Output:** The script will generate a submission file named `test_sub.csv` in the same directory, containing the predicted disease outbreak totals for the test set.  It also prints training progress and model evaluation information to the console.

This documentation provides a comprehensive overview of the Python script's functionality, structure, and usage. It explains each component of the code, enabling users to understand, modify, and utilize the script effectively for disease outbreak prediction.
"""
