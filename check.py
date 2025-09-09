import pandas as pd
import numpy as np

# load dataset safely
df = pd.read_csv('fdataset.csv', dtype={'ChronicConditions': str})

print("\n==== BASIC INFO ====")
print(df.info())

print("\n==== COLUMN HEADS ====")
print(df.columns.tolist())

print("\n==== FIRST 5 ROWS ====")
print(df.head().to_string(index=False))

print("\n==== MISSING VALUES ====")
missing = df.isna().sum()
print(missing[missing > 0])

print("\n==== UNIQUE VALUES FOR CHRONIC CONDITIONS ====")
print(df['ChronicConditions'].value_counts().head(20))

print("\n==== RISK SCORE STATS ====")
risk = df['Riskiness']
print(f"min: {risk.min():.3f}, max: {risk.max():.3f}, mean: {risk.mean():.3f}, std: {risk.std():.3f}")

print("\n==== DETERIORATION LABEL DISTRIBUTION ====")
print(df['Deterioration_within_90days'].value_counts())

print("\n==== DAYS PER PATIENT ====")
days_per_patient = df.groupby('PatientID')['Day'].max()
print(f"min: {days_per_patient.min()}, max: {days_per_patient.max()}, mean: {days_per_patient.mean():.2f}")

print("\n==== SAMPLE PATIENT CHRONIC CONDITIONS ====")
sample_patients = df.groupby('PatientID')['ChronicConditions'].first()
print(sample_patients.value_counts().head(10))

print("\n==== SUMMARY STATS FOR NUMERIC COLUMNS ====")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe().T)
