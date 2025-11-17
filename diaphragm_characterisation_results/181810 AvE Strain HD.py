#Test Characterisation:

# 250mL wateer increments for first 10 fills and then 250mL warm water

#To do:
# Add theoretical strain to the immediate below ✅
# Recorded Height vs Actual Height ✅
# Do the same for next dataset 
# Retreive temperature effect from data ✅

#Take a measure of plateaus, 

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.dates as mdates

#Dataframe Setup
df = pd.read_csv("20251003 181810-ILLumiSense-Strain-CH8 - Water added + Temperature Change.txt", sep="\t", header=38, encoding="cp1252")

#Helper Function
def theoretical_strain(rho, g, V, R, mu, E, t):
    return ((3*rho*g*(V/(math.pi))*(1-(mu**2)))/(8*E*(t**2)))*(10**6)

def absolute_error(experimental, expected):
    return abs(((expected-experimental)/experimental)*100)

#Calculation Constants
c = {
    "rho": 1000,  # Density in kg/m^3
    "g": 9.81,     # Gravitational acceleration in m/s^2
    "E": 2E11,  # Young's modulus in Pa
    "mu": 0.285,    # Poisson's ratio
    "t": 1.2E-4,      # Thickness in m
    "r": 0.10455,     # Radius in m
}

#Fill increment data
fill_volumes = [0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275] #cubic metres
fill_times = ["18:18:10","18:18:32", "18:19:01", "18:19:31", "18:20:02", "18:20:31", "18:21:01", "18:21:32", "18:22:01", "18:22:32", "18:23:35"]

#Expected Strain List
expected_strains = np.zeros(len(fill_volumes))

for i, volume in enumerate(fill_volumes):
    expected_strains[i] = theoretical_strain(c["rho"], c["g"], volume, c["r"], c["mu"], c["E"], c["t"])

anticipated_strains = np.zeros(len(df))

current_strain = 0
fill_index = 0

for i, time in enumerate(df["Time"]):
    if fill_index < len(fill_times) and time >= fill_times[fill_index]:
        current_strain = expected_strains[fill_index]
        fill_index += 1
    anticipated_strains[i] = current_strain

#Error List 
error = np.zeros(len(df))

for i,time in enumerate(df["Time"]):
    error[i] = absolute_error(df["0-26"][i], anticipated_strains[i])
    
#Time Formatting
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

#Tri Plot
# fig, ax1 = plt.subplots(figsize=(10,5))

# # --- Axis 1 (left)
# ax1.plot(df["Time"], df["0-26"], color="blue", label="Experimental")
# ax1.set_xlabel("Time")
# ax1.set_ylabel("MicroStrain", color="blue")
# ax1.tick_params(axis='y', colors="blue")

# # --- Axis 2 (right)
# ax2 = ax1.twinx()
# ax2.plot(df["Time"], anticipated_strains, color="green", linestyle="dashed", label="Expected")
# ax2.set_ylabel("Expected MicroStrain", color="green")
# ax2.tick_params(axis='y', colors="green")

# # --- Axis 3 (offset to the right)
# ax3 = ax1.twinx()                                # create another twin axis
# ax3.spines["right"].set_position(("outward", 60))  # move it 60 points to the right
# ax3.plot(df["Time"], error, color="red", label="Error (%)")
# ax3.set_ylabel("Error (%)", color="red")
# ax3.tick_params(axis='y', colors="red")

# # --- Formatting and aesthetics
# ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
# fig.autofmt_xdate()

# fig.suptitle("Experimental vs Actual Microstrain")
# ax1.grid(True)

# # Combine legends manually
# lines, labels = [], []
# for ax in [ax1, ax2, ax3]:
#     L = ax.get_legend_handles_labels()
#     lines += L[0]
#     labels += L[1]
# ax1.legend(lines, labels, loc="upper left")

# #Dual Plot
fig1, axis1 = plt.subplots(figsize=(10,5))

# --- Axis 1 (left)
axis1.plot(df["Time"], df["0-26"], color="blue", label="Experimental")
axis1.set_xlabel("Time")
axis1.set_ylabel("Microstrain", color="blue")
axis1.tick_params(axis='y', colors="blue")
axis1.plot(df["Time"], anticipated_strains, color="green", linestyle="dashed", label="Expected")

# --- Axis 3 (offset to the right)
axis2 = axis1.twinx()                                # create another twin axis
axis2.spines["right"].set_position(("outward", 60))  # move it 60 points to the right
axis2.plot(df["Time"], error, color="red", label="Error (%)")
axis2.set_ylabel("Error (%)", color="red")
axis2.tick_params(axis='y', colors="red")

# --- Formatting and aesthetics
axis1.xaxis.set_major_locator(mdates.AutoDateLocator())
axis1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
fig1.autofmt_xdate()
fig1.suptitle("Experimental vs Actual Microstrain")
axis1.grid(True)
fig1.legend(["Experimental","Expected", "Error"],loc='upper left')

#Show Plots
plt.show()



