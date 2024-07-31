import shap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load the cleaned dataset
startup_data_cleaned = pd.read_csv('data/startup_data_cleaned.csv')

# Define feature columns and target variable
feature_columns = ['age_first_funding_year', 'age_last_funding_year', 'relationships', 'funding_rounds',
                   'funding_total_usd', 'milestones', 'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate',
                   'is_software', 'is_web', 'is_mobile', 'is_enterprise', 'is_advertising', 'is_gamesvideo',
                   'is_ecommerce', 'is_biotech', 'is_consulting', 'is_othercategory', 'has_VC', 'has_angel',
                   'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD', 'avg_participants', 'is_top500',
                   'duration_of_operation', 'funding_duration', 'avg_funding_per_round']

X = startup_data_cleaned[feature_columns]
y = startup_data_cleaned['labels']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model with best parameters
best_rf = RandomForestClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=5, n_estimators=300, random_state=42)
best_rf.fit(X_train_scaled, y_train)

# Train the Gradient Boosting model with best parameters
best_gb = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, min_samples_leaf=1, min_samples_split=5, n_estimators=300, random_state=42)
best_gb.fit(X_train_scaled, y_train)

# Initialize the SHAP explainer for Random Forest
explainer_rf = shap.Explainer(best_rf, X_train_scaled)
shap_values_rf = explainer_rf(X_test_scaled, check_additivity=False)

# Initialize the SHAP explainer for Gradient Boosting
explainer_gb = shap.Explainer(best_gb, X_train_scaled)
shap_values_gb = explainer_gb(X_test_scaled, check_additivity=False)

# Ensure SHAP values are in the correct format
if hasattr(shap_values_rf, 'values'):
    shap_values_rf_array = shap_values_rf.values
else:
    shap_values_rf_array = shap_values_rf

if hasattr(shap_values_gb, 'values'):
    shap_values_gb_array = shap_values_gb.values
else:
    shap_values_gb_array = shap_values_gb

# Reshape SHAP values if necessary
if len(shap_values_rf_array.shape) == 3:
    shap_values_rf_array = shap_values_rf_array[:, :, 1]

if len(shap_values_gb_array.shape) == 3:
    shap_values_gb_array = shap_values_gb_array[:, :, 1]

# Plot SHAP summary plot for Random Forest
shap.summary_plot(shap_values_rf_array, X_test_scaled, feature_names=feature_columns)
plt.savefig('plots/shap_summary_rf.png')
plt.show()

# Plot SHAP summary plot for Gradient Boosting
shap.summary_plot(shap_values_gb_array, X_test_scaled, feature_names=feature_columns)
plt.savefig('plots/shap_summary_gb.png')
plt.show()
