import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta


#This file generates the data for the Toy-Dataset
# Number of rows to generate
num_rows = 50000

# Generate random dates for the fileID in 'YYYYMMDD' format
start_date = datetime(2020, 1, 1)
date_list = [(start_date + timedelta(days=i)).strftime("%Y%m%d") for i in range(num_rows)]
# Shuffle or just pick randomly
random_dates = [random.choice(date_list) for _ in range(num_rows)]

# Create a list of MRNs, some repeated
mrn_choices = list(range(1001, 1001 + 5000))  # MRNs from 1001 to 1500
mrns = [random.choice(mrn_choices) for _ in range(num_rows)]


# Generate dummy values
fileIDs = [f'ecg_{d}_id{i}' for i, d in enumerate(random_dates, start=1)]
QRS_Duration = np.random.uniform(60, 120, size=num_rows)     
QT_Interval = np.random.uniform(300, 500, size=num_rows)       
QTc_Interval = np.random.uniform(350, 450, size=num_rows)       
P_Axis = np.random.randint(-30, 75, size=num_rows)              
QRS_Axis = np.random.randint(-30, 180, size=num_rows)           
T_Axis = np.random.randint(-30, 90, size=num_rows)              
PatientSex_ECGData = np.random.choice(['M', 'F'], size=num_rows)
# Ages as strings like 'XXY'
ages = [f'{random.randint(20,60)}Y' for _ in range(num_rows)]
predictions_EF = np.random.rand(num_rows)  # Between 0 and 1

# Make Under40 random True/False
Under40 = [random.choice([True, False]) for _ in range(num_rows)]

# Create DataFrame
df = pd.DataFrame({
    'fileID': fileIDs,
    'MRN': mrns,
    'QRS_Duration': QRS_Duration,
    'QT_Interval': QT_Interval,
    'QTc_Interval': QTc_Interval,
    'P_Axis': P_Axis,
    'QRS_Axis': QRS_Axis,
    'T_Axis': T_Axis,
    'PatientSex_ECGData': PatientSex_ECGData,
    'PatientAge_ECGData': ages,
    'predictions_EF': predictions_EF,
    'Under40': Under40
})

# Save to CSV
df.to_csv('Toy_Dataset.csv', index=False)
print("Dummy data CSV generated with random Under40 values as True/False.")