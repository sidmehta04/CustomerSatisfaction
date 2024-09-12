import os
import pandas as pd
import glob

def combine_csv_files(existing_file, input_folder, output_file):
    # Read the existing CSV file
    existing_df = pd.read_csv(existing_file)
    
    # Get a list of all CSV files in the input folder
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    # Read and combine all CSV files
    combined_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    
    # Combine the existing CSV with the new data
    final_df = pd.concat([existing_df, combined_df], ignore_index=True)
    
    # Write the combined data to the output file
    final_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

# Example usage
existing_file = "data/archive/olist_customers_dataset.csv"
input_folder = "data/archive/"
output_file = "data/archive/combined.csv"

combine_csv_files(existing_file, input_folder, output_file)