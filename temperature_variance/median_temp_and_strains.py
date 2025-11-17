import matplotlib.pyplot as plt
import numpy as np
import math

"""0° C"""
# average_internal_temperature = [1.7
# 2.1
# 2.4
# 2.4
# 2.65
# 3.3
# 3.65
# 3.7]
# average_water_temperature = [-0.2
# 0.1
# 0.3
# 0.5
# 0.8
# 1.05
# 1.2 
# 1.7]
zero_c_median_strain_1 = [-165.4, -92.1, -160.5, -185.6, -195.3, -218.5, -237.7, -232.4]
zero_c_height_1 = [0.02832775, 0.05648375, 0.08475094, 0.11286621, 0.14086146, 0.1689437, 0.19725163, 0.22504761]
zero_c_median_strain_2 = [4.4, 101.4, 19.6, -22.7, -55.1, -79.9, -86.9, -92.4]
zero_c_height_2 = [0.02817141, 0.05639567, 0.08439093, 0.11255684, 0.14079981, 0.16896462, 0.19697419, 0.22507954]

plt.figure()
plt.plot(zero_c_median_strain_1, zero_c_height_1, label='0° C - Trial 1', color='blue')
plt.plot(zero_c_median_strain_2, zero_c_height_2, label='0° C - Trial 2', color='blue')

"""10° C"""
average_internal_temperature = [10.65, 10.9, 11, 11.1, 10.85, 10.85, 10.75, 10.85]
average_water_temperature = [9.9, 10.1, 10.2, 10.4, 10.5, 10.55, 10.55, 10.6]
ten_c_median_strain_1 = [-17.2, 9.5, -3.5, -3.1, 2.9, 10.4, 18, 26.1]
ten_c_height_1 = [0.0284582, 0.05755503, 0.0859947, 0.11442854, 0.1436349, 0.17246374, 0.20053987, 0.22943396]
ten_c_median_strain_2 = [-27.9, 21.6, -6, 17.6, -13.1, -6.7, 1.1, 11.7]
ten_c_height_2 = [0.02835461, 0.05661251, 0.08501956, 0.11345224, 0.14181967, 0.17013699, 0.1983646, 0.22664231]

plt.plot(ten_c_median_strain_1, ten_c_height_1, label='10° C - Trial 1', color='green')
plt.plot(ten_c_median_strain_2, ten_c_height_2, label='10° C - Trial 2', color='green')

"""20° C"""
# average_water_temperature = [19.325
# 19.2
# 19.225
# 19.225
# 19.15
# 19.175
# 19.275
# 19.3]
twenty_c_median_1 = [-26.8, 18.4, 33.6, 49.7, 65.8, 81.3, 96.3, 111.2]
twenty_c_height_1 = [0.0288106, 0.05733764, 0.08559511, 0.11420966, 0.1427402, 0.17117738, 0.19998682, 0.22859437] 
twenty_c_median_2 = [-5.8, 43, 56.2, 71.8, 87.9, 103.4, 118.6, 133.2]
twenty_c_height_2 = [0.02859822, 0.05691754, 0.08547608, 0.11414898, 0.14269586, 0.17132208 ,0.19979544, 0.22812643] 
twenty_c_median_3 = [-4.9, 38.2, 53.5, 69.8, 86.1, 101.5, 116.9, 131.5]
twenty_c_height_3 = [0.02862272, 0.05643559, 0.08494162, 0.11359585, 0.14208555, 0.1705624, 0.19900192, 0.22757213] 
twenty_c_median_4 = [-3.3, 35.3, 51.5, 68.4, 85.1, 100.9, 116.1, 130.7] 
twenty_c_height_4 = [0.02858188, 0.05708091, 0.08568846, 0.11432286, 0.14271803, 0.17118788, 0.20001366, 0.22849868] 

plt.plot(twenty_c_median_1, twenty_c_height_1, label='20° C - Trial 1', color='orange')
plt.plot(twenty_c_median_2, twenty_c_height_2, label='20° C - Trial 2', color='orange')
plt.plot(twenty_c_median_3, twenty_c_height_3, label='20° C - Trial 3', color='orange')
plt.plot(twenty_c_median_4, twenty_c_height_4, label='20° C - Trial 4', color='orange')

"""30° C"""
# average_internal_temperature = [29.95, 29.5, 29.2, 28.65, 28.75, 28.9, 29.05, 29.35]
# average_water_temperature = [31.5, 31.15, 31.05, 30.8, 30.6, 30.35, 29.9, 29.2]
thirty_c_median_1 = [-41.3, 113.2, 128.1, 133.9, 143.9, 147.3, 151.3, 156.2]
thirty_c_heights_1 = [0.02850863, 0.05679118, 0.08509354, 0.11344951, 0.14178567, 0.17040617, 0.19885653, 0.22757375]
thirty_c_median_2 = [-31.7, 11, 43.6, 61.7, 78.1, 89.3 ,97.7, 105.6]
thirty_c_heights_2 = [0.02861584, 0.05712097, 0.08526019, 0.11405665, 0.14249885, 0.17073013, 0.19948464, 0.22765649]

plt.plot(thirty_c_median_1, thirty_c_heights_1, label='30° C - Trial 1', color='red')
plt.plot(thirty_c_median_2, thirty_c_heights_2, label='30° C - Trial 2', color='red')

plt.xlabel("Strain (µε)", fontsize=12)
plt.ylabel("Height (m)", fontsize=12)
plt.title("Uncompensated Callibration Curves for Diaphragm FBG's at 0°C, 10°C, 20°C and 30°C", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

