import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import functions

rho = 998.19 #kg/m^3
r = 0.10455 #m

test_1_df = pd.read_csv('20251016 140537-ILLumiSense-Strain-CH8.txt', sep="\t", header=38, encoding="cp1252")
test_2_df = pd.read_csv('20251016 142822-ILLumiSense-Strain-CH8.txt', sep="\t", header = 38, encoding="cp1252")
test_3_df = pd.read_csv('20251016 144310-ILLumiSense-Strain-CH8.txt', sep="\t", header=38, encoding="cp1252")
test_4_df = pd.read_csv('20251016 145228-ILLumiSense-Strain-CH8.txt', sep="\t", header = 38, encoding="cp1252")

test1_pourtimes = [test_1_df["Time"][0], "14:06:00", "14:07:00", "14:08:00", "14:09:00", "14:10:00", "14:11:00", "14:12:00", "14:13:00"]

test2_pourtimes = [test_2_df["Time"][0], "14:29:00", "14:30:00", "14:31:00", "14:32:00", "14:33:00", "14:34:00", "14:35:00", "14:36:00"]

test3_pourtimes = [test_3_df["Time"][0], "14:43:30", "14:44:00", "14:44:30", "14:45:00", "14:45:30", "14:46:00", "14:47:00", "14:48:00"]

test4_pourtimes = [test_4_df["Time"][0], "14:53:00", "14:54:00", "14:55:00", "14:56:00", "14:57:00", "14:58:00", "14:59:00", "15:00:00"]

test_1_df = functions.convert_interogator_to_relative(test_1_df)
test_2_df = functions.convert_interogator_to_relative(test_2_df)
test_3_df = functions.convert_interogator_to_relative(test_3_df)
test_4_df = functions.convert_interogator_to_relative(test_4_df)


test1_pour_masses = np.array([0, 246.89, 244.46, 242.15, 245.21, 244.49, 243.69, 246.88, 245.15])
test2_pour_masses = np.array([0, 245.07, 242.68, 244.73, 245.71, 244.63, 245.31, 244, 242.78])
test3_pour_masses = np.array([0, 245.28, 238.34, 244.28, 245.55, 244.14, 244.03, 243.71, 244.83])
test4_pour_masses = np.array([0, 244.93, 244.22, 245.15, 245.38, 243.33, 243.97, 247.02, 244.1])

test1_pour_s = functions.pour_times_to_relative(test1_pourtimes)
test2_pour_s = functions.pour_times_to_relative(test2_pourtimes)
test3_pour_s = functions.pour_times_to_relative(test3_pourtimes)
test4_pour_s = functions.pour_times_to_relative(test4_pourtimes)

# Calculate cumulative poured mass
test1_pour_cum = np.cumsum(test1_pour_masses)
test2_pour_cum = np.cumsum(test2_pour_masses)
test3_pour_cum = np.cumsum(test3_pour_masses)
test4_pour_cum = np.cumsum(test4_pour_masses)

median_strains_1, median_times_1  = functions.median_strain_all_intervals(test_1_df["Time_s"].values, test_1_df["0-26"].values, test1_pour_s)
median_strains_2, median_times_2 = functions.median_strain_all_intervals(test_2_df["Time_s"].values, test_2_df["0-26"].values, test2_pour_s)
median_strains_3, median_times_3  = functions.median_strain_all_intervals(test_3_df["Time_s"].values, test_3_df["0-26"].values, test3_pour_s)
median_strains_4, median_times_4 = functions.median_strain_all_intervals(test_4_df["Time_s"].values, test_4_df["0-26"].values, test4_pour_s)

median_strains_temp_comp_1, median_strain_temp_comp_times_1 = functions.median_strain_all_intervals(test_1_df["Time_s"].values, test_1_df["0-26"].values-(test_1_df["0-20"].values-test_1_df["0-20"][0]), test1_pour_s)

median_strains_temp_comp_2, median_strain_temp_comp_times_2 = functions.median_strain_all_intervals(test_2_df["Time_s"].values, test_2_df["0-26"].values-(test_2_df["0-20"].values-test_2_df["0-20"][0]), test2_pour_s)

median_strains_temp_comp_3, median_strain_temp_comp_times_3 = functions.median_strain_all_intervals(test_3_df["Time_s"].values, test_3_df["0-26"].values-(test_3_df["0-20"].values-test_3_df["0-20"][0]), test3_pour_s)

median_strains_temp_comp_4, median_strain_temp_comp_times_4 = functions.median_strain_all_intervals(test_4_df["Time_s"].values, test_4_df["0-26"].values-(test_4_df["0-20"].values-test_4_df["0-20"][0]), test4_pour_s)

temp_max_strain_1_idx = test_1_df['0-20'].idxmax()
temp_max_strain_2_idx = test_2_df["0-20"].idxmax()
temp_max_strain_3_idx = test_3_df["0-20"].idxmax()
temp_max_strain_4_idx = test_4_df["0-20"].idxmax()

print(f"Trail 1 Maximum: {test_1_df['0-20'].max()} at {temp_max_strain_1_idx}")
print(f"Trail 2 Maximum: {test_2_df['0-20'].max()} at {temp_max_strain_2_idx}")
print(f"Trail 3 Maximum: {test_3_df['0-20'].max()} at {temp_max_strain_3_idx}")
print(f"Trail 4 Maximum: {test_4_df['0-20'].max()} at {temp_max_strain_4_idx}")

median_strains_temp_comp_1, median_strain_temp_comp_times_1 = functions.median_strain_all_intervals(test_1_df["Time_s"].values, test_1_df["0-26"].values-(test_1_df["0-20"].values-test_1_df["0-20"][temp_max_strain_1_idx]), test1_pour_s)
print(median_strains_temp_comp_1)
median_strains_temp_comp_2, median_strain_temp_comp_times_2 = functions.median_strain_all_intervals(test_2_df["Time_s"].values, test_2_df["0-26"].values-(test_2_df["0-20"].values-test_2_df["0-20"][temp_max_strain_2_idx]), test2_pour_s)
print(median_strains_temp_comp_2)
median_strains_temp_comp_3, median_strain_temp_comp_times_3 = functions.median_strain_all_intervals(test_3_df["Time_s"].values, test_3_df["0-26"].values-(test_3_df["0-20"].values-test_3_df["0-20"][temp_max_strain_3_idx]), test3_pour_s)
print(median_strains_temp_comp_3)
median_strains_temp_comp_4, median_strain_temp_comp_times_4 = functions.median_strain_all_intervals(test_4_df["Time_s"].values, test_4_df["0-26"].values-(test_4_df["0-20"].values-test_4_df["0-20"][temp_max_strain_4_idx]), test4_pour_s)
print(median_strains_temp_comp_4)


test1_heights = np.array([functions.find_true_height(mass, rho, r) for mass in test1_pour_cum])
test2_heights = np.array([functions.find_true_height(mass, rho, r) for mass in test2_pour_cum])
test3_heights = np.array([functions.find_true_height(mass, rho, r) for mass in test3_pour_cum])
test4_heights = np.array([functions.find_true_height(mass, rho, r) for mass in test4_pour_cum])

test1_strains = np.array(test_1_df['0-26'])
test2_strains = np.array(test_2_df['0-26'])
test3_strains = np.array(test_3_df['0-26'])
test4_strains = np.array(test_4_df['0-26'])

"""Median Strain and Liquid Level Height Measurement Plots"""
fig, ax1 = plt.subplots(figsize=(10,6))
# --- Left axis: Strain ---
ax1.plot(test1_heights[1:], median_strains_1, color="green", label="Test 1")
print(test1_heights, median_strains_1)
ax1.plot(test2_heights[1:], median_strains_2, color='orange', label="Test 2")
print(test2_heights, median_strains_2)
ax1.plot(test3_heights[1:], median_strains_3, color="black", label="Test 3")
print(test3_heights, median_strains_3)
ax1.plot(test4_heights[1:], median_strains_4, color='blue', label="Test 4")
print(test4_heights, median_strains_4)

# Add horizontal bars at median strain values
ax1.set_xlabel("Height (m)")
ax1.set_ylabel("Strain (µε)", color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax1.grid(True)

ax1.legend()

plt.title("Characterisation @ 20degC - Median Pour Interval Strain and True Height")
plt.tight_layout()
plt.show()


















