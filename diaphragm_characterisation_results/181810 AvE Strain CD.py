# Test Characterisation Script (Cleaned Up)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.dates as mdates

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(
    "20251003 181810-ILLumiSense-Strain-CH8 - Water added + Temperature Change.txt",
    sep="\t",
    header=38,
    encoding="cp1252"
)

# -----------------------------
# Helper Functions
# -----------------------------
def theoretical_strain(rho, g, V, R, mu, E, t):
    return ((3 * rho * g * (V / math.pi) * (1 - mu ** 2)) / (8 * E * t ** 2)) * 1e6

def absolute_error(experimental, expected):
    return abs((expected - experimental) / experimental * 100)

def composite_strain_at_fibre(q, R, layer_props, z_fibre_from_ref):
    A, z_centroids, Estar = [], [], []
    for i, layer in enumerate(layer_props):
        t = layer["t"]
        E = layer["E"]
        nu = layer["nu"]
        z_bottom = sum(lp["t"] for lp in layer_props[:i]) if i > 0 else 0.0
        zc = z_bottom + t / 2.0
        A.append(t)
        z_centroids.append(zc)
        Estar.append(E / (1 - nu ** 2))

    A, zc, Estar = np.array(A), np.array(z_centroids), np.array(Estar)
    z_na = np.sum(Estar * A * zc) / np.sum(Estar * A)
    Ics = np.array([(layer_props[i]["t"] ** 3) / 12.0 for i in range(len(layer_props))])
    D = np.sum(Estar * (Ics + A * (zc - z_na) ** 2))
    w0 = q * R ** 4 / (64.0 * D)
    eps_fibre = ((z_fibre_from_ref - z_na) * q * R ** 2) / (16.0 * D)
    return w0, eps_fibre

# -----------------------------
# Constants
# -----------------------------
c = {
    "rho": 1000,
    "g": 9.81,
    "E": 2e11,
    "mu": 0.285,
    "t": 1.2e-4,
    "r": 0.10455 / 2
}

# Layer Properties
layers = [
    {"t": 0.00012, "E": 200e9, "nu": 0.285},     # diaphragm
    {"t": 0.00046, "E": 3.4e9,  "nu": 0.35},     # top epoxy
    {"t": 0.00024735, "E": 37.5e9, "nu": 0.28},  # top GFRP
    {"t": 0.0000053, "E": 4.2e9, "nu": 0.05},    # cable
    {"t": 0.00024735, "E": 37.5e9, "nu": 0.28},  # bottom GFRP
    {"t": 0.00026, "E": 3.4e9,  "nu": 0.35}      # bottom epoxy
]
cable_pos_in_stack = 0.00046 + 0.0002235 + (0.0000053 / 2) + 0.00012

# -----------------------------
# Theoretical Strain Calculation
# -----------------------------
fill_volumes = np.array([0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125,
                         0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275])
fill_times = ["18:18:10","18:18:32","18:19:01","18:19:31","18:20:02",
              "18:20:31","18:21:01","18:21:32","18:22:01","18:22:32","18:23:35"]

expected_h = fill_volumes / (math.pi * c["r"] ** 2)
expected_strains = np.zeros(len(fill_volumes))

for i, volume in enumerate(fill_volumes):
    _, expected_strains[i] = composite_strain_at_fibre(c["rho"] * c["g"] * expected_h[i], c["r"], layers, cable_pos_in_stack)

anticipated_strains = np.zeros(len(df))
current_strain, fill_index = 0, 0

for i, time in enumerate(df["Time"]):
    if fill_index < len(fill_times) and time >= fill_times[fill_index]:
        current_strain = expected_strains[fill_index]
        fill_index += 1
    anticipated_strains[i] = current_strain

df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S")

# -----------------------------
# Plotting Function
# -----------------------------
def plot_channel(channel_name):
    error = np.zeros(len(df))
    for i in range(len(df)):
        error[i] = absolute_error(df[channel_name][i], anticipated_strains[i] * 1e7)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df["Time"], df[channel_name], color="black", label=f"Experimental")
    ax1.plot(df["Time"], anticipated_strains*10**7, color="green", linestyle="dashed", label="Expected")
    ax1.set_xlabel("Time (HH:MM:SS)", fontsize=20)
    ax1.set_ylabel("Strain (µε)", fontsize=20)
    ax1.tick_params(axis='both', colors="black", labelsize=20)

    ax2 = ax1.twinx()
    ax2.spines["right"]
    ax2.plot(df["Time"], error, color="red", label="Error (%)")
    ax2.set_ylabel("Error (%)", color="red", fontsize=20)
    ax2.tick_params(axis='y', colors="red", labelsize=20)

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.suptitle(f"Experimental vs Expected Strain (µε) for 100mL Water Increments", fontsize=20)
    ax1.grid(True)
    fig.legend(
    loc="center right",
    bbox_to_anchor=(0.8, 0.5),  # x just inside the right edge, y center
    fontsize=20,
    frameon=True
    )
    plt.tight_layout()
    plt.show()

# -----------------------------
# Plot Both Channels
# -----------------------------
print("Doing it")
plot_channel("0-26")
plot_channel("0-20")
plt.show()
