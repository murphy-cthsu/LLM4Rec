import pandas as pd

# Load the CSV file
file_path = "./sampled_data_with_predicted_class_0.csv"
df = pd.read_csv(file_path)

# Filter out rows where the 'Class' column is 'Failed Prediction'
cleaned_df = df[df['Class'] != 'Failed_Prediction']

# Save the cleaned DataFrame to a new CSV
output_path = "cleaned_data_class_0.csv"
cleaned_df.to_csv(output_path, index=False)