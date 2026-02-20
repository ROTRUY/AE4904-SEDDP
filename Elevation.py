### IMPORTS
import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

### FUNCTIONS
def process(Print: bool = True, save: bool = False, saveFolder: str = "NewFolder", stations:list[str]=None):
    """
    ...
    """
    
    # Folder containing the report folders
    folder_path = "GMATReports"

    # Get all csv files
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # Create list of average elevation above minimum and list of stations, corresponding to each other
    avg_elevations = list()

    for file in csv_files:
        # Get the name of the station the file is about
        original_name = os.path.basename(file)  # Get original name
        name_without_csv = os.path.splitext(original_name)[0]  # Remove ".csv"
        station_name = name_without_csv.replace("Report", "")  # Remove "Report"
        
        if not stations or station_name in stations:
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

            if save:
                # Build output path
                new_name = station_name + "Processed.csv"
                output_path = os.path.join(f"ProcessedReports\{saveFolder}", new_name)

                df.to_csv(output_path, index=False)

            # Get average elevation values, save to dict
            valid_elevations = df[df.Elevation >= 30]
            avg_elevations.append(np.average(valid_elevations.Elevation))

    avg_elevation = np.average(avg_elevations)
    if Print:
        print(avg_elevation)

### RUN HERE
process(False, True, "55deg")