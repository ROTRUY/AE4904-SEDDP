import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

# Folder containing csv files
folder_path = "GMATReports"

# Get all csv files
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Create list of average elevation above minimum and list of stations, corresponding to each other
avg_elevations = list()
stations = list()

for file in csv_files:
    print(f"Processing {file}...")
    
    # Read csv
    df = pd.read_csv(file)
    
    # Extract columns
    X = df.iloc[:, 4]
    Y = df.iloc[:, 5]
    Z = df.iloc[:, 6]
    
    # Compute slant range
    rho = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Compute elevation
    elevation = np.degrees(np.arctan2(Z, np.sqrt(X**2 + Y**2)))
    
    # Add columns
    df["SlantRange"] = rho
    df["Elevation"] = elevation
    
    # Build name for export
    original_name = os.path.basename(file)            # AvillaReport.csv
    name_without_csv = os.path.splitext(original_name)[0]  # AvillaReport
    
    # Remove "Report" if present and add "Processed"
    station_name = name_without_csv.replace("Report", "")
    new_name = station_name + "Processed.csv"
    
    # Full output path
    output_path = os.path.join("ProcessedReports", new_name)

    save = False

    if save:
        df.to_csv(output_path, index=False)

    # Get average elevation values, save to dict
    valid_elevations = df[df.Elevation >= 30]
    stations.append(station_name)
    avg_elevations.append(np.average(valid_elevations.Elevation))


avg_elevation = np.average(avg_elevations)

print(avg_elevation)
# AVG elevation: 43.99743011978639 deg