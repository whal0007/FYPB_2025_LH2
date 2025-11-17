import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.dates as mdates

#Dataframe Setup
df = pd.read_csv("20251003 181810-ILLumiSense-Strain-CH8 - Water added + Temperature Change.txt", sep="\t", header=38, encoding="cp1252")

#Helper Function
def theorectical_height():
    return 

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
fill_volumes = np.array([0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275]) #cubic metres
fill_times = np.array(["18:18:10","18:18:32", "18:19:01", "18:19:31", "18:20:02", "18:20:31", "18:21:01", "18:21:32", "18:22:01", "18:22:32", "18:23:35"])

#Expected Height List
expected_h = fill_volumes/(math.pi*(c["r"]**2))

#print(expected_h)

#Expected Height Filling 
heights = np.zeros(len(df))

current_height = 0
fill_index = 0

for i, time in enumerate(df["Time"]):
    if fill_index < len(fill_times) and time >= fill_times[fill_index]:
        current_height = expected_h[fill_index]
        fill_index += 1
    heights[i] = current_height

experimental_heights = np.array(8*c["E"]*(c["t"]**2)*(df["0-26"]/(10**6)))/(3*c["rho"]*c["g"]*(1-(c["mu"]**2))*(c["r"]**2))

# print(heights)
# print(experimental_heights)


# #Error List 
# error = np.zeros(len(df))

# for i,time in enumerate(df["Time"]):
#     error[i] = absolute_error(df["0-26"][i], anticipated_strains[i])

# error = np.array(abs(((heights-experimental_heights)/heights)*100))

error = np.zeros_like(heights, dtype=float)
np.divide(
    abs(heights - experimental_heights) * 100,
    heights,
    out=error,
    where=heights != 0
)


# plt.plot(df["Time"], heights)
# plt.plot(df["Time"], experimental_heights)
# plt.xlabel("Time")
# plt.ylabel("Height")

# fig1, axis1 = plt.subplots(figsize=(10,5))

# # --- Axis 1 (left)
# axis1.plot(df["Time"], experimental_heights, color="blue", label="Experimental")
# axis1.set_xlabel("Time")
# axis1.set_ylabel("Height", color="blue")
# axis1.tick_params(axis='y', colors="blue")
# axis1.plot(df["Time"], heights, color="green", linestyle="dashed", label="Expected")

# # --- Axis 3 (offset to the right)
# axis2 = axis1.twinx()                                # create another twin axis
# axis2.spines["right"].set_position(("outward", 60))  # move it 60 points to the right
# axis2.plot(df["Time"], error, color="red", label="Error (%)")
# axis2.set_ylabel("Error (%)", color="red")
# axis2.tick_params(axis='y', colors="red")

# # --- Formatting and aesthetics
# axis1.xaxis.set_major_locator(mdates.AutoDateLocator())
# axis1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
# fig1.autofmt_xdate()
# fig1.suptitle("Experimental vs Actual Height")
# axis1.grid(True)
# fig1.legend(["Experimental","Expected", "Error"],loc='upper left')

# Linearity Curve
plt.plot(df["0-26"], heights)

#Show Plots
plt.show()