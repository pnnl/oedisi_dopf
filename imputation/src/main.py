#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 09:08:13 2025

@author: mitr284
"""

import numpy as np
import os
import opendssdirect as dss
from opendssdirect.utils import Iterator
import argparse


def init_paths(args):
    if not os.path.exists(args.input):
        print("data location doesn't exist")
        exit()

    output = os.path.abspath(f"{args.output}/{args.model}")
    os.makedirs(output, exist_ok=True)
    return output


def init_opendss(args):
    os.chdir(os.path.abspath(args.input))

    dss.Text.Command("Clear")
    dss.Text.Command("Redirect master.dss")
    dss.Text.Command('Solve')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anonymize OpenDSS data.")
    parser.add_argument(
        "--model", help="model name: ieee123, SFO-P1U, ...")
    parser.add_argument(
        "--input", help="model path: ./opendss/master.dss")
    parser.add_argument(
        "--output", help="new model path: ./anon/opendss/")
    args = parser.parse_args()
    feeder_name = args.model
    input = args.input
    output = init_paths(args)
    init_opendss(args)

    # Simulation parameters
    n_runs = 100
    mults = np.linspace(0.1, 1.0, n_runs)
    steps = 4
    hrs = 1
    n_steps = steps * hrs

    load_names = dss.Loads.AllNames()
    dss.Solution.Solve()

    # Get buses
    bus_names = dss.Circuit.AllBusNames()
    n_buses = len(bus_names)
    max_phases = 3

    vol_v = []
    voltage_data = np.full((n_buses, max_phases), np.nan)
    voltage_data = np.nan_to_num(voltage_data)
    voltage_data_pv = np.full((n_buses, max_phases), np.nan)
    voltage_data_pv = np.nan_to_num(voltage_data_pv)
    load_data = np.full((n_buses, max_phases), np.nan)
    load_data = np.nan_to_num(load_data)
    load_data_tot = []
    vol_pv = []
    pv_total_kw = []
    base_kws = []

    pv_bus_list = []
    pv_name_list = []

    load_bus_list = []
    load_name_list = []
    load_total_kw = []

    pv_names = dss.PVsystems.AllNames()

    for pv_name in pv_names:
        # Set the active PV system
        dss.PVsystems.Name(pv_name)
        # Get bus name (bus1 property)
        pv_name_1 = dss.PVsystems.Name  # pv names
        dss.Circuit.SetActiveElement(f"PVsystem.{pv_name}")
        pv_bus = dss.CktElement.BusNames()[0]  # PV buses
        pv_bus_list.append(pv_bus)
        pv_name_list.append(pv_name_1)

    load_name = dss.Loads.AllNames()

    # for load in load_name:
    for load_name_1 in dss.Loads.AllNames():
        dss.Circuit.SetActiveElement(f"Load.{load_name_1}")
        load_bus = dss.CktElement.BusNames()[0]  # PV buses
        load_bus_list.append(load_bus)
        load_name_list.append(load_name_1)

    # Run simulation
    for i, mult in enumerate(mults):
        dss.Solution.Mode(0)
        dss.Solution.StepSize(1)
        dss.Solution.Hour(1)
        dss.Solution.ControlMode(0)
        # dss.Text.Command("Set mode=snap")
        # dss.Text.Command("Set stepsize=1h")
        # dss.Text.Command("Set number=1")
        # dss.Text.Command("Set controlmode=static")
        dss.Solution.LoadMult(mult)
        dss.Solution.Solve()

        # Record bus voltages
        for b, bus in enumerate(bus_names):
            dss.Circuit.SetActiveBus(bus)
            vmag = dss.Bus.VMagAngle()[::2]
            for p in range(min(len(vmag), max_phases)):
                voltage_data[b, p] = vmag[p]
        vol_v.append(voltage_data.copy())

        # LOAD

        load_snapshot = np.zeros((n_buses, max_phases))

        for name, bus in zip(load_name_list, load_bus_list):

            bus_eq = bus.split('.')[0]

            bus_idx = bus_names.index(bus_eq)

            powers = dss.Circuit.TotalPower()
            dss.Circuit.SetActiveElement(f"Load.{name}")
            phases = dss.CktElement.NumPhases()
            per_phase_kw = dss.Loads.kW() / phases

            if len(bus.split('.')) != 1:
                for i in range(phases):

                    cur_phase = int(bus.split('.')[i+1])

                    load_snapshot[bus_idx, cur_phase-1] = per_phase_kw

            elif phases == 3:

                for p in range(phases):

                    # load_snapshot[bus_idx, phase - 1] = per_phase_kw
                    load_snapshot[bus_idx, p] = per_phase_kw
        load_total_kw.append(load_snapshot.copy())

        # PV
        for name in pv_names:
            dss.Text.Command(f"Edit PVSystem.{name} Irradiance={mult}")
        dss.Solution.Solve()

        pv_snapshot = np.zeros((n_buses, max_phases))

        # FIX NEEDED SIMILAR TO LOAD

        for name, bus in zip(pv_name_list, pv_bus_list):

            bus_eq = bus.split('.')[0]

            if bus_eq in bus_names:
                bus_idx = bus_names.index(bus_eq)
            else:
                print(bus_eq)

            powers = dss.Circuit.TotalPower()
            dss.Circuit.SetActiveElement(f"PVSystem.{name}")
            phases = dss.CktElement.NumPhases()
            # Fill real power per phase (only if known; here assume equal split)
            per_phase_kw = dss.PVsystems.kW() / phases
            for p in range(phases):
                pv_snapshot[bus_idx, p] = per_phase_kw
        pv_total_kw.append(pv_snapshot.copy())

    vol_v = np.array(vol_v)
    pv_total_kw = np.array(pv_total_kw)
    load_total_kw = np.array(load_total_kw)

    print("Voltage shape:", vol_v.shape)
    print("PV injections (kW):", pv_total_kw.shape)
    print('Load data:', load_total_kw.shape)
    np.save(f"{output}/voltage.npy", vol_v)
    np.save(f"{output}/injection.npy", pv_total_kw)
    np.save(f"{output}/load.npy", load_total_kw)

    line_names = dss.Lines.AllNames()
    n_lines = len(line_names)
    max_phases = 3

    r_data = np.full((n_lines, max_phases, max_phases), np.nan)
    x_data = np.full((n_lines, max_phases, max_phases), np.nan)

    for i, line in enumerate(line_names):
        dss.Lines.Name(line)
        n_phases = dss.Lines.Phases()

        try:
            r_flat = dss.Lines.Rmatrix()
            x_flat = dss.Lines.Xmatrix()
            r_matrix = np.array(r_flat).reshape(n_phases, n_phases)
            x_matrix = np.array(x_flat).reshape(n_phases, n_phases)
        except Exception:

            r_matrix = np.eye(n_phases) * dss.Lines.R1()
            x_matrix = np.eye(n_phases) * dss.Lines.X1()

        # Pad to 3x3
        r_padded = np.full((max_phases, max_phases), np.nan)
        x_padded = np.full((max_phases, max_phases), np.nan)
        r_padded[:n_phases, :n_phases] = r_matrix
        x_padded[:n_phases, :n_phases] = x_matrix

        r_data[i] = r_padded
        x_data[i] = x_padded

    r_data = np.nan_to_num(r_data)
    x_data = np.nan_to_num(x_data)

    print(r_data.shape)
    print(x_data.shape)

    np.save(f"{output}/r.npy", r_data)
    np.save(f"{output}/x.npy", x_data)
