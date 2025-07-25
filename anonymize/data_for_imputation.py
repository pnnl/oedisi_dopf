#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 09:08:13 2025

@author: mitr284
"""

import numpy as np
import os, sys
import opendssdirect as dss


dss_folder = r'/Users/mitr284/Downloads/oedisi-ieee123-main/snapshot/'
os.chdir(dss_folder)
output_dir = os.path.join(dss_folder, "sim_output")
os.makedirs(output_dir, exist_ok=True)

n_runs = 100
mults = np.linspace(0.1,1.0,n_runs)
steps = 4
hrs = 1
n_steps = steps * hrs

dss.Basic.ClearAll()
dss.run_command("Redirect Master.dss")
dss.run_command("Compile Master.dss")
dss.Solution.Solve()


bus_names = dss.Circuit.AllBusNames()
n_buses = len(bus_names)
max_phases = 3

for i, mult in enumerate(mults):
    # 
    
    dss.Basic.ClearAll()
    dss.Text.Command('Redirect Master.dss')
    dss.Text.Command("Set mode=duty")
    dss.Text.Command("Set stepsize=0.25")
    dss.Text.Command("Set number=4")
    dss.Text.Command("Set controlmode=static")
    
    dss.Solution.LoadMult(mult)
    
    voltage_data = np.full((n_steps, n_buses, max_phases), np.nan)
    voltage_data = np.nan_to_num(voltage_data) #converting nan to 0
    
    for t in range(n_steps):
        dss.Solution.Solve()
        for b, bus in enumerate(bus_names):
            dss.Circuit.SetActiveBus(bus)
            vmag = dss.Bus.VMagAngle()[::2]  
            n_phases = len(vmag)
            for p in range(min(n_phases, max_phases)):
                voltage_data[t, b, p] = vmag[p]

    out_file = os.path.join(output_dir, f"voltages_mult_{mult:.2f}.npy")
    np.save(out_file, voltage_data)
    

# Resistance data

line_name = dss.Lines.AllNames()
n_len = len(line_name)
n_steps = 4

resistance_data = np.full((n_steps, n_len, max_phases, max_phases), np.nan)

for t in range(n_steps):
    dss.Solution.Solve()
    for l_idx, line in enumerate(line_name):
        dss.Lines.Name(line)
        n_phases = dss.Lines.Phases()
        try:
            rflat = dss.Lines.Rmatrix()
            r_matrix = np.array(rflat).reshape((n_phases, n_phases))
        except Exception:
            
            r_matrix = np.eye(n_phases) * dss.Lines.R1()

        # Pad to 3x3 with NaNs
        padded = np.full((max_phases, max_phases), np.nan)
        padded[:n_phases, :n_phases] = r_matrix
        resistance_data[t, l_idx] = padded

resistance_data = np.nan_to_num(resistance_data) #converting nan to 0
resistance_data = resistance_data[1,:]

np.save(output_dir+"//resistance_data.npy", resistance_data)


#Reactance data


line_name = dss.Lines.AllNames()
n_len = len(line_name)
n_steps = 4

resistance_data = np.full((n_steps, n_len, max_phases, max_phases), np.nan)

for t in range(n_steps):
    dss.Solution.Solve()
    for l_idx, line in enumerate(line_name):
        dss.Lines.Name(line)
        n_phases = dss.Lines.Phases()
        try:
            rflat = dss.Lines.Xmatrix()
            r_matrix = np.array(rflat).reshape((n_phases, n_phases))
        except Exception:
            
            r_matrix = np.eye(n_phases) * dss.Lines.R1()

        # Pad to 3x3 with NaNs
        padded = np.full((max_phases, max_phases), np.nan)
        padded[:n_phases, :n_phases] = r_matrix
        resistance_data[t, l_idx] = padded

resistance_data = np.nan_to_num(resistance_data) #converting nan to 0
resistance_data = resistance_data[1,:]

np.save(output_dir+"//reactance_data.npy", resistance_data)


#PV multiplier data


for i, mult in enumerate(mults):
    # 
    
    dss.Basic.ClearAll()
    dss.Text.Command('Redirect Master.dss')
    dss.Text.Command("Set mode=duty")
    dss.Text.Command("Set stepsize=0.25")
    dss.Text.Command("Set number=4")
    dss.Text.Command("Set controlmode=static")
    
    gen_names = dss.Generators.AllNames()
    kws =[]
    
    for gen_name in gen_names:
        dss.Generators.Name(gen_name)
        kws.append(dss.Generators.kW())
    
    for gen_name, base_kw in zip(gen_names, kws):
        dss.Generators.Name(gen_name)
        dss.Generators.kW(base_kw * mult)
    
    bus_names = dss.Circuit.AllBusNames()
    n_buses = len(bus_names)
    pv_data = np.full((n_steps, n_buses, max_phases), np.nan)
    pv_data =np.nan_to_num(pv_data)

    for t in range(n_steps):
        dss.Solution.Solve()
        for b, bus in enumerate(bus_names):
            dss.Circuit.SetActiveBus(bus)
            vmag = dss.Bus.VMagAngle()[::2]
            for p in range(min(len(vmag), max_phases)):
                pv_data[t, b, p] = vmag[p]
    out_file = os.path.join(output_dir, f"voltages_gen_mult_{mult:.2f}.npy")
    np.save(out_file, pv_data)
            