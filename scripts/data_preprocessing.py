import pandas as pd

# Load the dataset
#The data set is found on kaggle at this adress : https://www.kaggle.com/datasets/manishkc06/startup-success-prediction
file_path = 'data/startup_data.csv'
startup_data = pd.read_csv(file_path)

# Drop irrelevant or mostly empty columns
columns_to_drop = ['Unnamed: 0', 'Unnamed: 6', 'id', 'zip_code', 'city', 'state_code.1', 'object_id']
startup_data_cleaned = startup_data.drop(columns=columns_to_drop)

# Convert date columns to datetime objects
date_columns = ['founded_at', 'closed_at', 'first_funding_at', 'last_funding_at']
for col in date_columns:
    startup_data_cleaned[col] = pd.to_datetime(startup_data_cleaned[col], errors='coerce')

# Handle missing values
startup_data_cleaned['closed_at'] = startup_data_cleaned['closed_at'].fillna(pd.Timestamp('2099-12-31'))
startup_data_cleaned['age_first_milestone_year'] = startup_data_cleaned['age_first_milestone_year'].fillna(startup_data_cleaned['age_first_milestone_year'].median())
startup_data_cleaned['age_last_milestone_year'] = startup_data_cleaned['age_last_milestone_year'].fillna(startup_data_cleaned['age_last_milestone_year'].median())

# Feature Engineering
startup_data_cleaned['duration_of_operation'] = (startup_data_cleaned['closed_at'] - startup_data_cleaned['founded_at']).dt.days
startup_data_cleaned['funding_duration'] = (startup_data_cleaned['last_funding_at'] - startup_data_cleaned['first_funding_at']).dt.days
startup_data_cleaned['avg_funding_per_round'] = startup_data_cleaned['funding_total_usd'] / startup_data_cleaned['funding_rounds']
startup_data_cleaned['avg_funding_per_round'] = startup_data_cleaned['avg_funding_per_round'].fillna(0)

# Save the cleaned dataset
startup_data_cleaned.to_csv('data/startup_data_cleaned.csv', index=False)
