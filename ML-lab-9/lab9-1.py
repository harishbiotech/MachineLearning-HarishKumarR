#Write a program to partition a dataset (simulated data for regression)
# into two parts, based on a feature (BP) and for a threshold, t = 80.
# Generate additional two partitioned datasets based on different threshold values of t = [78, 82].
import pandas as pd
df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
df_bp_80=df.sort_values(by='BP',ascending=True)
df_bp_above_80=df_bp_80[df_bp_80['BP']>80]
df_bp_below_80=df_bp_80[df_bp_80['BP']<=80]
print(df_bp_above_80)
print("--------------------------------------------")
print(df_bp_below_80)