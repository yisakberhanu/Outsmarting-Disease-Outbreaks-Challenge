# Mastering Disease Outbreak Prediction with Time Series Forecasting

The **SUA Challenge on Zindi** was more than just a data science competition; it was a vital initiative focused on predicting disease outbreaks, leveraging the power of AI for social good.  This challenge, one of two winning projects supported by the **AI For Equity Challenge** in partnership with **IRCAI** and **AWS**, utilized data from the **Amazon Sustainability Data Initiative (ASDI)** – Amazon’s tech-for-good program providing crucial scientific data for sustainability research, including climate change.  The solutions developed in this challenge, including my winning approach detailed in this blog post, will directly support the **Sokoine University of Agriculture (SUA)** in Tanzania, empowering them to implement and deploy these forecasting tools. Hosted on the **Zindi** platform, this competition exemplified how **Amazon Web Services (AWS)**, with its world-leading cloud technologies, and the **International Research Centre on Artificial Intelligence (IRCAI)**, with its focus on AI for Sustainable Development Goals, are collaborating to drive impactful solutions for global challenges, benefiting institutions like **Sokoine University of Agriculture (SUA)** and communities worldwide. For participants like myself, it was an inspiring opportunity to contribute to a project with real-world consequences, applying time series forecasting expertise to improve lives through disease outbreak prediction.

# **Technical Foundation:**

**Feature Engineering:** 

Feature engineering is paramount in time series forecasting.  I transformed raw data into meaningful features that capture temporal patterns, environmental influences, and disease-specific dynamics. Let's explore some key feature engineering techniques:

**1. Temporal Feature Engineering:**

Temporal feature engineering is about encoding the time aspect of our data in a way that machine learning models can understand.  I started by converting date components (Year, Month) into a datetime object, allowing me to extract time-based information. I also create features like `time_index` (numerical timestamp) and `diff_date` (days since the start of observation for each location) to represent the progression of time.

**2. Lag Feature Creation:**

Lag features are crucial for time series forecasting as they incorporate past values of the target variable.  They allow the model to learn from historical patterns and seasonality. Here's a code snippet demonstrating lag feature creation:

```python
import pandas as pd

def create_lag_features(df, target_column, lag_periods):
    """
    Creates lag features for a given target column in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
        lag_periods (list): List of lag periods in months (e.g., [1, 2, 3]).

    Returns:
        pd.DataFrame: DataFrame with added lag features.
    """
    df_with_lags = df.copy()
    date_col = df_with_lags[['date']] # Assuming 'date' column exists

    for lag in lag_periods:
        shifted_date = date_col + pd.DateOffset(months=lag)
        lag_df = df_with_lags[[target_column, 'Disease', 'date', 'Location', 'tag_id']].copy() # Selecting relevant cols
        lag_df['date'] = shifted_date
        lag_df.rename(columns={target_column: f'{target_column}_lag_{lag}'}, inplace=True) # Descriptive lag column name
        df_with_lags = pd.merge(df_with_lags, lag_df, on=['Disease', 'date', 'Location', 'tag_id'], how='left') # Clear merge keys
    return df_with_lags

# Example usage (assuming 'train' DataFrame and 'Total' target column):
lag_months = range(1, 36) # Lags up to 35 months
train_sum_with_lags = create_lag_features(train_sum, 'Total', lag_months)
```

This function, `create_lag_features`, takes a DataFrame, the target column name, and a list of lag periods (in months). It iterates through the lag periods, shifts the date column, and merges the shifted target values back into the original DataFrame, creating new columns like `Total_lag_1`, `Total_lag_2`, etc. These lag features represent the historical 'Total' disease outbreaks from previous months.

**3. Static Feature Generation:**

To condense the information from multiple lag features, we can generate static features by calculating statistical summaries across a range of lag columns. This can provide a more robust and generalized representation of past temporal trends. Here’s an example function:

```python
import pandas as pd

def create_static_features(df, feature_columns):
    """
    Creates static features by summarizing a list of feature columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_columns (list): List of feature column names to summarize.

    Returns:
        pd.DataFrame: DataFrame with added static features.
    """
    df_static_feats = df.copy()
    summary_cols = feature_columns # Columns to summarize
    if not summary_cols: # Handle empty list case
        return df_static_feats # Return original if no cols to summarize

    df_static_feats['median_cases'] = df_static_feats[summary_cols].median(axis=1)
    df_static_feats['mean_cases'] = df_static_feats[summary_cols].mean(axis=1)
    df_static_feats['max_cases'] = df_static_feats[summary_cols].max(axis=1)
    df_static_feats['sum_cases'] = df_static_feats[summary_cols].sum(axis=1)
    df_static_feats['std_cases'] = df_static_feats[summary_cols].std(axis=1)
    # Add more static features as needed (skew, kurtosis, quantiles etc.)

    return df_static_feats

# Example usage (assuming 'train_sum_lagged' DataFrame and lag feature columns):
lag_feature_cols = [col for col in train_sum_lagged.columns if 'Total_lag_' in col] # List of lag columns
train_sum_static_feats = create_static_features(train_sum_lagged, lag_feature_cols)
```

The `create_static_features` function calculates statistics like median, mean, max, sum, and standard deviation across a list of specified feature columns (typically lag features). These static features capture the central tendency, magnitude, and variability of past disease outbreaks, providing aggregated temporal information to the model.

**4. Incorporating Environmental Data:**

Disease outbreaks are often influenced by environmental factors. We can integrate environmental datasets (like climate data, water source locations, sanitation data) by merging them with our main dataset based on location and time keys (e.g., Latitude, Longitude, Year, Month). This allows the model to consider environmental conditions when making predictions.

1. **Model Training and Prediction:**

      **A. Cross-Validation for Time Series:**

In my winning **SUA Challenge** solution, I strategically leveraged *both* **TimeSeriesSplit** and **KFold cross-validation**, and importantly, **ensembled their predictions** to achieve a more robust and potentially more accurate final forecast.  This might seem unconventional for time series at first, so let me clarify my approach and rationale:

- **TimeSeriesSplit: Primary Method for Temporal Forecasting Evaluation:** As repeatedly emphasized, **TimeSeriesSplit was the *core* and *primary* validation method for evaluating the time series forecasting performance** in the **SUA Challenge**. Its forward-chaining nature, training on past and validating on future data, provided the most reliable measure of how well my models would generalize to unseen future outbreaks – the key objective of the competition. I used TimeSeriesSplit to rigorously assess the temporal accuracy of my models and to tune hyperparameters specifically for time-dependent forecasting.
- **KFold: Capturing Broader Data Patterns and Diversity for Ensembling:** While TimeSeriesSplit focused on temporal accuracy, I also incorporated **KFold cross-validation to capture a broader range of patterns in the data and to create a more diverse ensemble.** Even though KFold shuffles temporal order and is not ideal for *direct* time series validation, it offered valuable benefits when used *in conjunction* with TimeSeriesSplit, particularly for ensembling:

**B. Target Transformations:**

In some cases, transforming the target variable can improve model performance. We explore a few target transformations:

- **Difference Target:** Predicting the *difference* in disease outbreaks from the previous time step. This can help in detrending the time series and making it more stationary.
- **Ratio Target:** Predicting the *ratio* of disease outbreaks compared to the previous time step. Useful if the target variable scales proportionally to its past values.
- **Original Target:** Predicting the raw, untransformed 'Total' disease outbreak count directly.

By training models on these different target transformations and ensembling their predictions, we can potentially capture different aspects of the underlying time series patterns and improve the robustness of our forecasts.

**C. Disease-Specific Modeling:**

It can be beneficial to adopt a disease-specific modeling approach. Different diseases might have different drivers and temporal patterns.  As illustrated in the code example, we can train separate models or use different modeling strategies (feature sets, model types, target transformations) for different diseases (e.g., Diarrhea, Malaria, Intestinal Worms). This allows for a more nuanced and potentially more accurate forecasting system.

**Ensemble Prediction:**  

To further boost the performance and stability of my **SUA Challenge** predictions, I employed ensemble methods.  Averaging predictions from models trained with different target transformations *and potentially also across different cross-validation strategies (TimeSeriesSplit and KFold)*, was a key ensembling technique in my winning solution.

Here's the simple averaging ensemble function I used for the **SUA Challenge** (which could be expanded to average predictions from TimeSeriesSplit and KFold runs):

```python
def ensemble_predictions_average(predictions_list):
    """
    Ensembles predictions by simple averaging.

    Args:
        predictions_list (list): List of prediction arrays (e.g., from different models).

    Returns:
        np.ndarray: Averaged prediction array.
    """
    return np.mean(predictions_list, axis=0)

# Example usage for the SUA Challenge (example of ensembling, could be expanded to include KFold predictions):
predictions_diff_target = test['Predicted_Total_bdr_diff'].values
predictions_ratio_target = test['Predicted_Total_bdr_ratio'].values
predictions_original_target = test['Predicted_Total_bdr_org'].values

ensemble_preds = ensemble_predictions_average([predictions_diff_target, predictions_ratio_target, predictions_original_target])
test['Ensemble_Predicted_Total'] = ensemble_preds
```

This `ensemble_predictions_average` function allowed me to combine the strengths of different models in my **SUA Challenge** solution, leading to more reliable and accurate final predictions, *and in my winning approach, this ensembling potentially included predictions from both TimeSeriesSplit and KFold based validation runs*.

The **SUA Challenge on Zindi** was a truly rewarding experience, allowing me to apply data science to address a critical global health issue. This blog post has outlined my winning approach, emphasizing the crucial roles of meticulous feature engineering, disease-specific modeling, strategic target transformations, robust cross-validation utilizing both TimeSeriesSplit and KFold ensembling, and carefully crafted disease-specific prediction adjustments.  By mastering and thoughtfully combining these techniques, as demonstrated in my SUA Challenge solution, you too can effectively tackle complex disease outbreak prediction problems, contribute meaningfully to proactive public health efforts, and achieve success in challenging competitions like the SUA Challenge, ultimately working towards a healthier future for communities worldwide.