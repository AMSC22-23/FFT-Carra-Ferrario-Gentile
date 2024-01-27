import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import subprocess
import sys
from datetime import datetime
import numpy as np

def convert_to_float(value):
    if value.endswith(' ms'):
        return float(value.replace(' ms', ''))
    elif value.endswith(' s'):
        return float(value.replace(' s', '')) * 1000
    elif value.endswith(' us'):
        return float(value.replace(' us', '')) / 1000
    else:
        return None
 
testname = sys.argv[1]
header1 = sys.argv[2]
header2 = sys.argv[3]
dim=int(sys.argv[4])
if dim==1:
    # Run the benchmark script and capture its output
    csv_data = subprocess.check_output(['../speedup/benchmark.sh  ../../build/' + testname + ' ' + header1 + ' ' + header2], shell=True)
elif dim==2:
    csv_data = subprocess.check_output(['../speedup/benchmark2D.sh  ../../build/' + testname + ' ' + header1 + ' ' + header2], shell=True)
else:
    csv_data = subprocess.check_output(['../speedup/benchmark3D.sh  ../../build/' + testname + ' ' + header1 + ' ' + header2], shell=True)


#print(csv_data)

# Read the data into a pandas DataFrame
data = pd.read_csv(StringIO(csv_data.decode('utf-8')))

# Convert time columns to numeric values (strip ' ms' and convert to float)
time_columns = [header1 + ' forward', header1 + ' inverse', header2 + ' forward', header2 + ' inverse']
for col in time_columns:
    data[col] = data[col].apply(convert_to_float)

# Drop rows with None values (non-convertible strings)
data.dropna(subset=time_columns, inplace=True)

# Plotting
plt.figure(figsize=(10, 5))

# Plot each implementation's forward and inverse times
for col in time_columns:
    plt.plot(data['n'], data[col], label=col)

# Calculate and plot n*log(n) line
n_values = data['n']
n_log_n_values = 2**(dim*n_values) * (dim*n_values)
#scale it 
n_log_n_values = n_log_n_values / n_log_n_values[0] * data[time_columns[3]][0]
plt.plot(n_values, n_log_n_values, label='n*log(n)', linestyle='--') 

# Labeling the plot
plt.xlabel('n (2^n FFT length)')
plt.ylabel('Time (ms)')
plt.title('FFT Implementations Performance')
plt.legend()
plt.grid(True)

plt.yscale('log')
    
 
# Save plot to PNG
plt.savefig(sys.argv[5])
