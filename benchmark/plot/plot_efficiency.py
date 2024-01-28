import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import subprocess
import sys
from datetime import datetime
import numpy as np
 
testname = sys.argv[1]
max_proc = sys.argv[2]
dim=int(sys.argv[3])
filename=sys.argv[4]

if dim==1:
    # Run the benchmark script and capture its output
    size = sys.argv[5]
    csv_data = subprocess.check_output(['../efficiency/efficiency.sh  ../../build/test/' + testname + ' ' +size + ' ' + max_proc], shell=True)
elif dim==2:
    size = sys.argv[5]
    size2 = sys.argv[6]
    csv_data = subprocess.check_output(['../efficiency/efficiency_2D.sh  ../../build/test/' + testname + ' ' +size + ' ' +size2+ ' ' + max_proc], shell=True)
else:
    size = sys.argv[5]
    size2 = sys.argv[6]
    size3 = sys.argv[7]

    csv_data = subprocess.check_output(['../efficiency/efficiency_3D.sh   ../../build/test/' + testname + ' ' +size + ' ' +size2 + ' ' +size3+ ' ' + max_proc], shell=True)


#print(csv_data)

# Read the data into a pandas DataFrame
data = pd.read_csv(StringIO(csv_data.decode('utf-8')))

print(data)

# Convert time columns to numeric values (strip ' ms' and convert to float)
y_col = ['SPEEDUP_F', 'SPEEDUP_B', 'EFFICIENCY_F', 'EFFICIENCY_B']
#for col in y_col:
#    data[col] = data[col].apply(convert_to_float)

# Drop rows with None values (non-convertible strings)
data.dropna(subset=y_col, inplace=True)

# Plotting
plt.figure(figsize=(10, 5))

# Plot each implementation's forward and inverse times
for col in y_col:
    plt.plot(data['p'], data[col], label=col)

# Labeling the plot
plt.xlabel('processes')
plt.ylabel('Efficiency and speedup')
plt.title('FFT Efficiency and Speedup')
plt.legend()
plt.grid(True)

#plt.yscale('log')
    
 
# Save plot to PNG
plt.savefig(filename)
