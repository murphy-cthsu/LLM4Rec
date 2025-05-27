import pandas as pd

# Load the CSV file
for i in range (10):
    file_path = f"sampled_data_with_predicted_class_{i}.csv"
    df = pd.read_csv(file_path)

    # Filter out rows where the 'Class' column is 'Failed Prediction'
    cleaned_df = df[df['Class'] != 'Failed_Prediction']

    # Save the cleaned DataFrame to a new CSV
    output_path = f"cleaned_data_class_{i}.csv"
    cleaned_df.to_csv(output_path, index=False)