import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.dates as mdates

rho = 999.70 #kg/m^3 at 10 deg C
r = 0.10455 #m

c = {
    "rho": 1000,  # Density in kg/m^3
    "g": 9.81,     # Gravitational acceleration in m/s^2
    "E": 2E11,  # Young's modulus in Pa
    "mu": 0.285,    # Poisson's ratio
    "t": 1.2E-4,      # Thickness in m
    "r": 0.10455,     # Radius in m
}

def median_strain_between_pours(data_time_s, strain_values, pour_times_s, pour_idx):
    """
    Compute the median strain value and the middle time
    between two consecutive pour times.

    Parameters
    ----------
    data_time_s : array-like
        Time values (in seconds) for the experimental data.
    strain_values : array-like
        Corresponding strain values.
    pour_times_s : array-like
        Pour times in seconds.
    pour_idx : int
        Index of the starting pour (median computed between pour_idx and pour_idx+1).

    Returns
    -------
    float
        Median strain value between the two pours.
    float
        Middle time (in seconds) between the two pours.
    """
    t_start = pour_times_s[pour_idx]
    t_end = pour_times_s[pour_idx + 1]
    t_mid = t_start + (t_end - t_start) / 2

    # Mask data between pours
    mask = (data_time_s >= t_start) & (data_time_s <= t_end)
    segment = strain_values[mask]

    if len(segment) == 0:
        return np.nan, t_mid
    else:
        return np.median(segment), t_mid

def median_strain_all_intervals(data_time_s, strain_values, pour_times_s):
    medians = []
    median_times = []
    for i in range(len(pour_times_s) - 1):
        med,times = median_strain_between_pours(data_time_s, strain_values, pour_times_s, i)
        medians.append(med)
        median_times.append(times)
    return np.array(medians), np.array(median_times)

def convert_interogator_to_relative(df, time_col="Time", new_col="Time_s"):
    """
    Convert a HH:MM:SS time column into seconds relative to the first timestamp.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the time column.
    time_col : str, optional
        Name of the column containing HH:MM:SS timestamps. Default is "Time".
    new_col : str, optional
        Name of the new column to store relative seconds. Default is "Time_s".
    
    Returns
    -------
    pandas.DataFrame
        The same DataFrame with added timedelta and relative time columns.
    """
    df["Time_deltaS"] = pd.to_timedelta(df[time_col])
    df[new_col] = df["Time_deltaS"].dt.total_seconds() - df["Time_deltaS"].dt.total_seconds().iloc[0]
    return df

def pour_times_to_relative(pour_times):
    """
    Convert a list of HH:MM:SS pour times to seconds relative 
    to the first element in the list (experiment start time).

    Parameters
    ----------
    pour_times : list of str
        List of timestamps (HH:MM:SS), where the first element
        is the experiment start time.

    Returns
    -------
    np.ndarray
        Array of pour times in seconds relative to experiment start.
    """
    td = pd.to_timedelta(pour_times)
    rel_seconds = td.total_seconds() - td.total_seconds()[0]
    return rel_seconds

test_1_df = pd.read_csv('20251016 152953-ILLumiSense-Strain-CH8.txt', sep="\t", header=38, encoding="cp1252")
test_2_df = pd.read_csv('20251016 164913-ILLumiSense-Strain-CH8.txt', sep="\t", header = 38, encoding="cp1252")
test_3_df = pd.read_csv('20251016 175933-ILLumiSense-Strain-CH8.txt', sep="\t", header=38, encoding="cp1252")

# Convert pour times
test_1_df = convert_interogator_to_relative(test_1_df)
test_2_df = convert_interogator_to_relative(test_2_df)
test_3_df = convert_interogator_to_relative(test_3_df)

test1_pourtimes = [test_1_df["Time"][0], "15:30:45", "15:32:38", "15:34:30", "15:37:05", "15:38:55", "15:40:45", "15:42:30", "15:44:30"]
test2_pourtimes = [test_2_df["Time"][0],"16:54:45", "16:57:10", "16:59:05", "17:02:25", "17:03:45", "17:05:35", "17:08:10", "17:10:05"]
test3_pourtimes = [test_3_df["Time"][0], "18:01:01", "18:03:28" ,"18:04:56", "18:07:22", "18:09:44", "18:11:35", "18:14:13", "18:15:58"]

test1_pour_masses = np.array([0, 244.239, 249.72, 244.08, 244.03, 250.66, 247.42, 240.96, 247.98])
test2_pour_masses = np.array([0, 243.35, 242.52, 243.8, 244.02, 243.46, 243.03, 242.26, 242.69])
test3_pour_masses = np.array([0, 243.55, 250.1, 243.25, 243.07, 241.87, 244.07, 240.78, 245.89])

start_time_test1 = test_1_df["Time"].iloc[0]

test1_pour_s = pour_times_to_relative(test1_pourtimes)
test2_pour_s = pour_times_to_relative(test2_pourtimes)
test3_pour_s = pour_times_to_relative(test3_pourtimes)

# Calculate cumulative poured mass
test1_pour_cum = np.cumsum(test1_pour_masses)
test2_pour_cum = np.cumsum(test2_pour_masses)
test3_pour_cum = np.cumsum(test3_pour_masses)

# test3_temperature = 

median_strains_1, median_times_1  = median_strain_all_intervals(test_1_df["Time_s"].values, test_1_df["0-26"].values, test1_pour_s)
# print("Medians for Test 1 are:")
# print(np.array(median_strains_1))
median_strains_3, median_times_3  = median_strain_all_intervals(test_3_df["Time_s"].values, test_3_df["0-26"].values, test3_pour_s)
# print("Medians for Test 3 are:")
# print(np.array(median_strains_3))

temp_max_strain_1_idx = test_1_df['0-20'].idxmax()
temp_max_strain_3_idx = test_3_df["0-20"].idxmax()

print(f"Trail 1 Maximum: {test_1_df['0-20'].max()} at {temp_max_strain_1_idx}")
print(f"Trail 2 Maximum: {test_3_df['0-20'].max()} at {temp_max_strain_3_idx}")

median_strains_temp_comp_1, median_strain_temp_comp_times_1 = median_strain_all_intervals(test_1_df["Time_s"].values, test_1_df["0-26"].values-(test_1_df["0-20"].values-test_1_df["0-20"][temp_max_strain_1_idx]), test1_pour_s)
print(median_strains_temp_comp_1)
median_strains_temp_comp_3, median_strain_temp_comp_times_3 = median_strain_all_intervals(test_3_df["Time_s"].values, test_3_df["0-26"].values-(test_3_df["0-20"].values-test_3_df["0-20"][temp_max_strain_3_idx]), test3_pour_s)
print(median_strains_temp_comp_3)

def find_true_height(mass, rho, r):
    height = (mass/1000)/(rho*math.pi*math.pow(r/2,2))
    return height

test1_experimental_heights = np.array(8*c["E"]*(c["t"]**2)*(test_1_df["0-26"]/(10**6)))/(3*c["rho"]*c["g"]*(1-(c["mu"]**2))*(c["r"]**2))
test2_experimental_heights = np.array(8*c["E"]*(c["t"]**2)*(test_2_df["0-26"]/(10**6)))/(3*c["rho"]*c["g"]*(1-(c["mu"]**2))*(c["r"]**2))
test3_experimental_heights = np.array(8*c["E"]*(c["t"]**2)*(test_3_df["0-26"]/(10**6)))/(3*c["rho"]*c["g"]*(1-(c["mu"]**2))*(c["r"]**2))

test1_heights = np.array([find_true_height(mass, rho, r) for mass in test1_pour_cum])
print("Test 1 Heights")
print(np.array(test1_heights))
test2_heights = np.array([find_true_height(mass, rho, r) for mass in test2_pour_cum])
test3_heights = np.array([find_true_height(mass, rho, r) for mass in test3_pour_cum])
print("Test 2 Heights")
print(np.array(test2_heights))

test1_strains = np.array(test_1_df['0-26'])
test2_strains = np.array(test_2_df['0-26'])
test3_strains = np.array(test_3_df['0-26'])

"""Top Boiler Plate"""
plt.figure()

""" Basic Plot with Outlier """
# plt.plot(test_1_df["Time_s"], test_1_df["0-26"], label="Test 1: Strange Response")
# plt.plot(test_2_df["Time_s"], test_2_df["0-26"], label="Test 2: Oven Vibration")
# plt.plot(test_3_df["Time_s"], test_3_df["0-26"], label="Test 3: No Vibration at Step Change")

""" Basic Plot without Outlier """
# plt.plot(test_1_df["Time_s"], test_1_df["0-26"], label="Test 1: Strange Response")
plt.plot(test_3_df["Time_s"][4500:], test_3_df["0-26"][4500:], label="DFBG")
plt.plot(test_3_df["Time_s"][4500:], test_3_df["0-20"][4500:], label="TFBG")

""" Initial Pour without Outlier Plot """
# plt.plot(test_1_df["Time_s"][:2000], test_1_df["0-26"][:2000], label="Test 1: Strange Response")
# plt.plot(test_3_df["Time_s"][:2000], test_3_df["0-26"][:2000], label="Test 3: No Vibration at Step Change")

"""Basic Plot of Outlier"""
# plt.plot(test_2_df["Time_s"], test_2_df["0-26"], label="Test 2: Oven Vibration")

"""Outlier Pour of Outlier"""
# plt.plot(test_2_df["Time_s"][:3000], test_2_df["0-26"][:3000], label="Test 2: Oven Vibration")

"""Simple cumaltive mass in cylinder and strain"""
# plt.plot(test_2_df["Time_s"], test_2_df["0-26"])
# plt.plot(test2_pour_s, test2_pour_cum, color="orange", label="Test 2 Cumulative mass", zorder=5)

"""Plot Boiler Plate"""
plt.ylabel("Strain (µε)", fontsize=20)
plt.xlabel("Time (s)", fontsize=20)
plt.title("Sensor Characterisation at 10 deg", fontsize=20)
plt.tick_params(axis="both", labelsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()

"""Test 1 - Strange Strain"""
fig, ax1 = plt.subplots(figsize=(10,6))

# --- Left axis: Strain ---
ax1.plot(test_1_df["Time_s"], test_1_df["0-26"], color="green", label="DFBG")
#ax1.plot(test_1_df["Time_s"], test_1_df["0-20"], color="blue", label="Fluid Submerged")
# ax1.plot(test_1_df["Time_s"], test_1_df["0-5"])
ax1.plot(test_1_df["Time_s"], test_1_df["0-20"], color='red', label="TFBG") #-test_1_df["0-20"][temp_max_strain_1_idx]
ax1.plot(test_1_df["Time_s"], test_1_df["0-26"]-(test_1_df["0-20"]-test_1_df["0-20"][temp_max_strain_1_idx]), color='fuchsia', label="DFBG - TFBG")
# ax1.plot(median_times_1, median_strains_1, color="red", label="Test 1 Interval Median Strain")

# Add horizontal bars at median strain values
# for i in range(len(median_strains_1)):
#     t_start = test1_pour_s[i]
#     t_end = test1_pour_s[i + 1]
#     y = median_strains_1[i]
#     ax1.hlines(y, t_start, t_end, colors="orange", linestyles="--", linewidth=2, label="Median strain" if i == 0 else "")

for i in range(len(median_strains_temp_comp_1)):
    t_start = test1_pour_s[i]
    t_end = test1_pour_s[i + 1]
    y = median_strains_temp_comp_1[i]
    ax1.hlines(y, t_start, t_end, colors="black", linestyles="--", linewidth=2, label="DFBG - TFBG Median" if i == 0 else "")


ax1.set_xlabel("Time (s)", fontsize=20)
ax1.set_ylabel("Strain (µε)", color="black", fontsize=20)
ax1.tick_params(axis="y", labelcolor="black", labelsize=20)
ax1.tick_params(axis="x", labelcolor="black", labelsize=20)
ax1.set_xlim(0,(test_1_df["LineNumber"].iloc[-1]-test_1_df["LineNumber"].iloc[0])/10)
ax1.grid(True)

# --- Right axis: Cumulative mass ---
ax2 = ax1.twinx()  # create secondary y-axis
ax2.plot(test1_pour_s, test1_heights, color="blue", linestyle="dashed", label="Liquid Level (m)")
ax2.set_ylabel("Height (m)", color="black", fontsize=20)
ax2.tick_params(axis="y", labelcolor="black", labelsize=20)

# Combine legends from both axes
strain_lines, strain_labels = ax1.get_legend_handles_labels()
mass_lines, mass_labels = ax2.get_legend_handles_labels()
ax1.legend(strain_lines + mass_lines, strain_labels + mass_labels, loc="lower right", fontsize=20)

plt.title("Experimental Characterisation at 10°C - Temperature Compensation (Trial 1)", fontsize=20)
plt.tight_layout()
plt.show()

# """Test 2 - Outlier"""
# fig, ax1 = plt.subplots(figsize=(10,6))

# # --- Left axis: Strain ---
# ax1.plot(test_2_df["Time_s"], test_2_df["0-26"], color="green", label="Test 2 Strain")
# ax1.set_xlabel("Time (s)")
# ax1.set_ylabel("Strain (µε)", color="black")
# ax1.tick_params(axis="y", labelcolor="black")
# ax1.grid(True)

# # # --- Right axis: Cumulative mass ---
# ax2 = ax1.twinx()  # create secondary y-axis
# ax2.plot(test2_pour_s, test2_pour_cum, color="orange", marker="o", linestyle="-", label="Test 2 Cumulative mass")
# ax2.set_ylabel("Cumulative mass (g)", color="black")
# ax2.tick_params(axis="y", labelcolor="black")

# # # Combine legends from both axes
# strain_lines, strain_labels = ax1.get_legend_handles_labels()
# mass_lines, mass_labels = ax2.get_legend_handles_labels()
# ax1.legend(strain_lines + mass_lines, strain_labels + mass_labels, loc="upper left")

# plt.title("Trial 2 - Characterisation @ 10degC")
# plt.tight_layout()
# plt.show()

"""Test 3 - Oven Off Between Intervals"""
# fig2, ax1 = plt.subplots(figsize=(10,6))

# # --- Left axis: Strain ---
# ax1.plot(test_3_df["Time_s"], test_3_df["0-26"], color="green", label="Test 2 Strain")
# ax1.plot(test_3_df["Time_s"], test_3_df["0-20"], color="blue", label="Test 2 Temperature Sensor")
# ax1.plot(median_times_3, median_strains_3, color="red", label="Test 2 Interval Median Strain")

# # Add horizontal bars at median strain values
# for i in range(len(median_strains_3)):
#     t_start = test3_pour_s[i]
#     t_end = test3_pour_s[i + 1]
#     y = median_strains_3[i]
#     ax1.hlines(y, t_start, t_end, colors="orange", linestyles="--", linewidth=2, label="Median strain" if i == 0 else "")
# ax1.set_xlabel("Time (s)")
# ax1.set_ylabel("Strain (µε)", color="black")
# ax1.tick_params(axis="y", labelcolor="black")
# ax1.grid(True)

# # --- Right axis: Cumulative mass ---
# ax2 = ax1.twinx()  # create secondary y-axis
# ax2.plot(test3_pour_s, test3_pour_cum, color="orange", marker="o", linestyle="-", label="Test 2 Cumulative mass")
# ax2.set_ylabel("Cumulative mass (g)", color="black")
# ax2.tick_params(axis="y", labelcolor="black")

# # Combine legends from both axes
# strain_lines, strain_labels = ax1.get_legend_handles_labels()
# mass_lines, mass_labels = ax2.get_legend_handles_labels()
# ax1.legend(strain_lines + mass_lines, strain_labels + mass_labels, loc="upper left")

# plt.title("Trial 2 - Characterisation at 10° C")
# plt.tight_layout()
# plt.show()

"""Median Strain and Liquid Level Height Measurement Plots"""
# fig3, ax1 = plt.subplots(figsize=(10,6))
# # --- Left axis: Strain ---
# ax1.plot(test1_heights[1:], median_strains_1, color="green", label="Test 1")
# ax1.plot(test3_heights[1:], median_strains_3, color='orange', label="Test 3")

# # Add horizontal bars at median strain values
# ax1.set_xlabel("Height (m)")
# ax1.set_ylabel("Strain (µε)", color="black")
# ax1.tick_params(axis="y", labelcolor="black")
# ax1.grid(True)

# ax1.legend()

# plt.title("Characterisation at 10°C - Median Pour Interval Strain and True Height")
# plt.tight_layout()
# plt.show()









