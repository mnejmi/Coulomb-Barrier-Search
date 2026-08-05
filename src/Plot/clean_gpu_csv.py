import pandas as pd
import sys
csv_file = sys.argv[1]
print('Reading:', csv_file)
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()
print('Columns:', df.columns.tolist())
time_col = None
for col in df.columns:
    if 'time' in col.lower():
        time_col = col
        break
if time_col is None:
    raise ValueError(' No timestamp column found!')
print('Using time column:', time_col)
for col in df.columns:
    if col != time_col:
        df[col] = df[col].astype(str).str.extract('([\\d\\.]+)').astype(float)
df[time_col] = pd.to_datetime(df[time_col])
df['time_sec'] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()
clean_file = csv_file.replace('.csv', '_clean.csv')
df.to_csv(clean_file, index=False)
print(' Clean CSV saved:', clean_file)