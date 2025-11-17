import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

df = pd.read_csv("20251003 181810-ILLumiSense-Strain-CH8 - Water added + Temperature Change.txt", sep="\t", header=38, encoding="cp1252")

temp_comp_normalised = np.array(df["0-20"] - df["0-20"][0])

plt.plot(df["Time"], df["0-26"], label="Measured Strain (με)", color='blue')
plt.plot(df["Time"], df["0-20"], label="Measured Strain On Temp Gauge", color='red')


plt.plot(df["Time"], df["0-20"]-df["0-20"][0], label="Normalised Temp (degC)", color='black')

plt.plot(df["Time"], df["0-26"]-(df["0-20"]-df["0-20"][0]), label="Temp Compensated Measured Strain (με)", color='green')
plt.xlabel("Time")
plt.title("Strain and Temperature Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()