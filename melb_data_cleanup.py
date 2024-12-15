import pandas as pd

# Load the Melbourne housing data
file_path = "melb_data.csv"  # Replace with your file name
output_file = "clean_melb_data.csv"

# Step 1: Load the data
df = pd.read_csv(file_path)

# Step 2: Check record size before cleanup
initial_size = df.shape[0]

# Step 3: Remove rows where Bedroom2 is missing (NaN) or 0
df_cleaned = df.dropna(subset=['Bedroom2'])  # Drop NaN values
df_cleaned = df_cleaned[df_cleaned['Bedroom2'] > 0]  # Keep only positive values

# Step 4: Check record size after cleanup
final_size = df_cleaned.shape[0]

# Step 5: Save cleaned data
df_cleaned.to_csv(output_file, index=False)

# print(f"Initial record size: {initial_size}")
# print(f"Final record size: {final_size}")
