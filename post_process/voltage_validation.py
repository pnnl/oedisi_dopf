import matplotlib.pyplot as plt
import json
import numpy as np

file_path_opf = 'solutions//voltage_opf.json'
file_path_opds = 'solutions//voltage_opendss.json'

# Read the JSON file and load data into a dictionary
with open(file_path_opf, 'r') as json_file:
    voltage_opf = json.load(json_file)

with open(file_path_opds, 'r') as json_file:
    voltage_opds = json.load(json_file)

v_dif={}
for key, val in voltage_opf.items():
    v_opf = np.array(voltage_opf[key])
    v_opds = np.array(voltage_opds[key])
    v_dif[key] = 100*((v_opf-v_opds)/v_opds)


Va = v_dif['Va']
Vb = v_dif['Vb']
Vc = v_dif['Vc']

fig, ax1 = plt.subplots(1)
ax1.scatter(range(1, len(Va) + 1), Va, c='blue', edgecolor='blue')
ax1.scatter(range(1, len(Vb) + 1), Vb, c='green', edgecolor='green')
ax1.scatter(range(1, len(Vc) + 1), Vc, c='red', edgecolor='red')
ax1.set_xlabel('Bus Indices', fontsize=12)
ax1.set_ylabel('Voltage Error (%)', fontsize=12)

ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

plt.legend(['Phase-A', 'Phase-B', 'Phase-C'])
plt.show()