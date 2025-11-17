import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.dates as mdates

rho = 999.57 #kg/m^3 at 30 deg C
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

test_1_df = pd.read_csv('20251016 194528-ILLumiSense-Strain-CH8.txt', sep="\t", header=38, encoding="cp1252")
test_2_df = pd.read_csv('20251016 204052-ILLumiSense-Strain-CH8.txt', sep="\t", header = 38, encoding="cp1252")


# Convert pour times
test_1_df = convert_interogator_to_relative(test_1_df)
test_2_df = convert_interogator_to_relative(test_2_df)

test1_pourtimes = [test_1_df["Time"][0], "19:46:20", "19:47:35", "19:48:55", "19:50:22", "19:51:50", "19:53:08", "19:54:27", "19:55:47"]
test2_pourtimes = [test_2_df["Time"][0], "20:41:16", "20:42:38", "20:43:52", "20:44:53", "20:46:19", "20:47:30", "20:48:48", "20:50:08"]

test1_env_temp = [29.7, 28.8, 28.8, 28.1, 28.4, 28.8, 28.8, 28.9]
test1_water_temp = [32.4, 31.7, 31.8, 31.4, 31.2, 30.8, 30.4, 29.3]

test2_env_temp = [30.2, 30.2, 29.6, 29.2, 29.1, 29.0, 29.3, 29.8]
test2_water_temp = [30.6, 30.6, 30.3, 30.2, 30.0, 29.9, 29.4, 29.1]


test1_pour_masses = np.array([0, 244.64, 242.7, 242.87, 243.33, 243.16, 245.6, 244.14, 246.43])
test2_pour_masses = np.array([0, 245.56, 244.61, 241.47, 247.11, 244.07, 242.26, 246.75, 241.75])

test1_pour_s = pour_times_to_relative(test1_pourtimes)
test2_pour_s = pour_times_to_relative(test2_pourtimes)

# Calculate cumulative poured mass
test1_pour_cum = np.cumsum(test1_pour_masses)
test2_pour_cum = np.cumsum(test2_pour_masses)

# test3_temperature = 

median_strains_1, median_times_1  = median_strain_all_intervals(test_1_df["Time_s"].values, test_1_df["0-26"].values, test1_pour_s)
print("Medians for Test 1 are:")
print(median_strains_1)
median_strains_2, median_times_2 = median_strain_all_intervals(test_2_df["Time_s"].values, test_2_df["0-26"].values, test2_pour_s)
print("Medians for Test 2 are:")
print(median_strains_2)

temp_max_strain_1_idx = test_1_df['0-20'].idxmax()
temp_max_strain_2_idx = test_2_df["0-20"].idxmax()

print(f"Trail 1 Maximum: {test_1_df['0-20'].max()} at {temp_max_strain_1_idx}")
print(f"Trail 2 Maximum: {test_2_df['0-20'].max()} at {temp_max_strain_2_idx}")

median_strains_temp_comp_1, median_strain_temp_comp_times_1 = median_strain_all_intervals(test_1_df["Time_s"].values, test_1_df["0-26"].values-(test_1_df["0-20"].values-test_1_df["0-20"][temp_max_strain_1_idx]), test1_pour_s)
print(median_strains_temp_comp_1)
median_strains_temp_comp_2, median_strain_temp_comp_times_2 = median_strain_all_intervals(test_2_df["Time_s"].values, test_2_df["0-26"].values-(test_2_df["0-20"].values-test_2_df["0-20"][temp_max_strain_2_idx]), test2_pour_s)
print(median_strains_temp_comp_2)

def find_true_height(mass, rho, r):
    height = (mass/1000)/(rho*math.pi*math.pow(r/2,2))
    return height

test1_experimental_heights = np.array(8*c["E"]*(c["t"]**2)*(test_1_df["0-26"]/(10**6)))/(3*c["rho"]*c["g"]*(1-(c["mu"]**2))*(c["r"]**2))
test2_experimental_heights = np.array(8*c["E"]*(c["t"]**2)*(test_2_df["0-26"]/(10**6)))/(3*c["rho"]*c["g"]*(1-(c["mu"]**2))*(c["r"]**2))

test1_heights = np.array([find_true_height(mass, rho, r) for mass in test1_pour_cum])
print(test1_heights)
test2_heights = np.array([find_true_height(mass, rho, r) for mass in test2_pour_cum])
print(test2_heights)


test1_strains = np.array(test_1_df['0-26'])
test2_strains = np.array(test_2_df['0-26'])


"""Top Boiler Plate"""
# plt.figure()

""" Basic Plot with Outlier """
# plt.plot(test_1_df["Time_s"], test_1_df["0-26"], label="Test 1: Strange Response")
# plt.plot(test_2_df["Time_s"], test_2_df["0-26"], label="Test 2: Oven Vibration")
# plt.plot(test_3_df["Time_s"], test_3_df["0-26"], label="Test 3: No Vibration at Step Change")

""" Basic Plot without Outlier """
# plt.plot(test_1_df["Time_s"], test_1_df["0-26"], label="Test 1: Strange Response")
# plt.plot(test_3_df["Time_s"], test_3_df["0-26"], label="Test 3: No Vibration at Step Change")

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
# plt.ylabel("Strain (µε)")
# plt.xlabel("Time (s)")
# plt.title("Sensor Characterisation @ 10 deg C w/ Outlier")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

"""Test 1 - Strange Strain"""
# fig, ax1 = plt.subplots(figsize=(10,6))

# # --- Left axis: Strain ---
# ax1.plot(test_1_df["Time_s"], test_1_df["0-26"], color="green", label="Test 1 Strain")
# ax1.plot(median_times_1, median_strains_1, color="red", label="Test 1 Interval Median Strain")

# # Add horizontal bars at median strain values
# for i in range(len(median_strains_1)):
#     t_start = test1_pour_s[i]
#     t_end = test1_pour_s[i + 1]
#     y = median_strains_1[i]
#     ax1.hlines(y, t_start, t_end, colors="orange", linestyles="--", linewidth=2, label="Median strain" if i == 0 else "")

# ax1.set_xlabel("Time (s)")
# ax1.set_ylabel("Strain (µε)", color="black")
# ax1.tick_params(axis="y", labelcolor="black")
# ax1.grid(True)

# # --- Right axis: Cumulative mass ---
# ax2 = ax1.twinx()  # create secondary y-axis
# ax2.plot(test1_pour_s, test1_pour_cum, color="black", marker="o", linestyle="-", label="Test 1 Cumulative mass")
# ax2.set_ylabel("Cumulative mass (g)", color="black")
# ax2.tick_params(axis="y", labelcolor="black")

# # Combine legends from both axes
# strain_lines, strain_labels = ax1.get_legend_handles_labels()
# mass_lines, mass_labels = ax2.get_legend_handles_labels()
# ax1.legend(strain_lines + mass_lines, strain_labels + mass_labels, loc="upper left")

# plt.title("Trial 1 - Characterisation @ 10degC")
# plt.tight_layout()
# plt.show()

"""Test 2 - Outlier"""
# fig, ax1 = plt.subplots(figsize=(10,6))

# # --- Left axis: Strain ---
# ax1.plot(test_2_df["Time_s"][6000:], test_2_df["0-26"][:], color="green", label="Test 2 Strain")
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
# fig, ax1 = plt.subplots(figsize=(10,6))

# # --- Left axis: Strain ---
# ax1.plot(test_3_df["Time_s"], test_3_df["0-26"], color="green", label="Test 3 Strain")
# ax1.plot(test_3_df["Time_s"], test_3_df["0-20"], color="blue", label="Test 3 Temperature Sensor")
# ax1.plot(median_times_3, median_strains_3, color="red", label="Test 3 Interval Median Strain")

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
# ax2.plot(test3_pour_s, test3_pour_cum, color="orange", marker="o", linestyle="-", label="Test 3 Cumulative mass")
# ax2.set_ylabel("Cumulative mass (g)", color="black")
# ax2.tick_params(axis="y", labelcolor="black")

# # Combine legends from both axes
# strain_lines, strain_labels = ax1.get_legend_handles_labels()
# mass_lines, mass_labels = ax2.get_legend_handles_labels()
# ax1.legend(strain_lines + mass_lines, strain_labels + mass_labels, loc="upper left")

# plt.title("Trial 3 - Characterisation @ 10degC")
# plt.tight_layout()
# plt.show()

"""Median Strain and Liquid Level Height Measurement Plots"""
# fig, ax1 = plt.subplots(figsize=(10,6))
# # --- Left axis: Strain ---
# ax1.plot(test1_heights[1:], median_strains_1, color="green", label="Test 1")
# ax1.plot(test2_heights[1:], median_strains_2, color='orange', label="Test 3")

# # Add horizontal bars at median strain values
# ax1.set_xlabel("Height (m)")
# ax1.set_ylabel("Strain (µε)", color="black")
# ax1.tick_params(axis="y", labelcolor="black")
# ax1.grid(True)

# ax1.legend()

# plt.title("Characterisation @ 10degC - Median Pour Interval Strain and True Height")
# plt.tight_layout()
# plt.show()


"""Showing all experimental data per test 0 deg"""
fig, (ax1,ax3) = plt.subplots(2,1,figsize=(10,6))

plt.suptitle("Experimental Characterisation at 30°C (Trial 2)", fontsize=20)

# --- Left axis: Strain ---
ax1.plot(test_2_df["Time_s"], test_2_df["0-26"], color="green", label="Diaphragm FBG")
ax2 = ax1.twinx()  # create secondary y-axis
ax2.plot(test2_pour_s, test2_heights, color="blue", label="Liquid Level (m)", linestyle='dashed')
ax2.set_ylabel("Height (m)", color="black", fontsize=20)
ax2.tick_params(axis="y", labelcolor="black")

# Add horizontal bars at median strain values
for i in range(len(median_strains_2)):
    t_start = test2_pour_s[i]
    t_end = test2_pour_s[i + 1]
    y = median_strains_2[i]
    ax1.hlines(y, t_start, t_end, colors="orange", linestyles="--", linewidth=2, label="Diaphragm FBG Median" if i == 0 else "")

# ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Strain (µε)", color="black", fontsize=20)
ax1.tick_params(axis="y", labelcolor="black", labelsize=20)
ax1.tick_params(axis="x", labelcolor="black", labelsize=20)
ax1.grid(True)
ax1.set_xlim(0,(test_2_df["LineNumber"].iloc[-1]-test_2_df["LineNumber"].iloc[0])/10)


ax3.plot(test_2_df["Time_s"], test_2_df["0-20"], color="red", label="Temp FBG")
ax3.tick_params(axis="y", labelcolor="black", labelsize=20)
ax3.tick_params(axis="x", labelcolor="black", labelsize=20)
ax3.set_ylabel("Strain (µε)", color="black", fontsize=20)
ax3.set_xlabel("Time (s)", fontsize=20)
ax3.set_xlim(0,(test_2_df["LineNumber"].iloc[-1]-test_2_df["LineNumber"].iloc[0])/10)
ax3.grid(True)
ax4 = ax3.twinx()                                # create another twin axis
ax4.spines["right"]  # move it 60 points to the right
ax4.plot(test2_pour_s[1:], test2_env_temp, color="black", label="Chamber (°C)")
ax4.plot(test2_pour_s[1:], test2_water_temp, color="grey", label="Water (°C)")
ax4.set_ylabel("Temperature (°C)", fontsize=20)

# Combine legends from both axes
strain_lines, strain_labels = ax1.get_legend_handles_labels()
height_lines, height_labels = ax2.get_legend_handles_labels()
tstrain_lines, tstrain_labels = ax3.get_legend_handles_labels()
temp_lines, temp_labels = ax4.get_legend_handles_labels()


ax1.legend(strain_lines + height_lines, strain_labels + height_labels, loc="center right", fontsize=20)
ax3.legend(tstrain_lines + temp_lines, tstrain_labels+ temp_labels, loc = "upper right", fontsize=20)

plt.tight_layout()
plt.show()









