import yaml
import pandas as pd
with open('configuration.yaml', 'r') as file:
    config = yaml.safe_load(file)
rows = []
for category, params in config.items():
    for param, value in params.items():
        rows.append({'Category': category, 'Parameter': param, 'Value': value})
df = pd.DataFrame(rows)
df.to_excel('logs/config.xlsx', index=True)
print(" Excel file 'config.xlsx' created successfully!")