import matplotlib.pyplot as plt
import numpy as np
import math

import numpy as np

def normalize_to_minus1_1(arr):
    """
    Normalize an array to the range [-1, 1].

    Parameters
    ----------
    arr : array-like
        Input list or NumPy array.

    Returns
    -------
    np.ndarray
        Normalized array with values between -1 and 1.
    """
    arr = np.array(arr, dtype=float)
    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val == min_val:
        # avoid division by zero — return zeros if all values are equal
        return np.zeros_like(arr)

    # scale to [0, 1]
    norm_0_1 = (arr - min_val) / (max_val - min_val)

    # scale to [-1, 1]
    norm_minus1_1 = norm_0_1 * 2 - 1
    return norm_minus1_1


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



zero_c_median_strain_1 = [ -49.3,  -52.9, -113.9, -136.4, -142.3, -167.2, -184.7, -178.5]
zero_c_height_1 = [0.02832775, 0.05648375, 0.08475094, 0.11286621, 0.14086146, 0.1689437, 0.19725163, 0.22504761]
zero_c_median_strain_2 = [123.9, 110.2,  41.2,   0.3, -31.6, -57.3, -64.9, -71.7]
zero_c_height_2 = [0.02817141, 0.05639567, 0.08439093, 0.11255684, 0.14079981, 0.16896462, 0.19697419, 0.22507954]

plt.figure()
plt.plot(normalize_to_minus1_1(zero_c_median_strain_1), zero_c_height_1, label='0° C - Trial 1', color='blue')
plt.plot(normalize_to_minus1_1(zero_c_median_strain_2), zero_c_height_2, label='0° C - Trial 2', color='blue')

"""10° C"""
# average_internal_temperature = [10.65, 10.9, 11, 11.1, 10.85, 10.85, 10.75, 10.85]
# average_water_temperature = [9.9, 10.1, 10.2, 10.4, 10.5, 10.55, 10.55, 10.6]
ten_c_median_strain_1 = [10.1, 11.7,  5.1, 7.8, 12.6, 20.6, 28.6, 36.7]
ten_c_height_1 = [0.0284582, 0.05755503, 0.0859947, 0.11442854, 0.1436349, 0.17246374, 0.20053987, 0.22943396]
ten_c_median_strain_2 = [80.,  36.6, 11.7,  5.1, 10.7, 19.5, 28.5, 39.9]
ten_c_height_2 = [0.02835461, 0.05661251, 0.08501956, 0.11345224, 0.14181967, 0.17013699, 0.1983646, 0.22664231]

plt.plot(normalize_to_minus1_1(ten_c_median_strain_1), ten_c_height_1, label='10° C - Trial 1', color='green')
plt.plot(normalize_to_minus1_1(ten_c_median_strain_2), ten_c_height_2, label='10° C - Trial 2', color='green')

"""20° C"""
# average_water_temperature = [19.325
# 19.2
# 19.225
# 19.225
# 19.15
# 19.175
# 19.275
# 19.3]
twenty_c_median_1 = [ 21.6,  21.,   36.3,  52.2,  68.3,  83.7, 98.6, 112.9]
twenty_c_height_1 = [0.0288106, 0.05733764, 0.08559511, 0.11420966, 0.1427402, 0.17117738, 0.19998682, 0.22859437] 
twenty_c_median_2 = [ 17.5,  44.8,  59.2,  75.,   91.1, 106.5, 121.5, 136.3]
twenty_c_height_2 = [0.02859822, 0.05691754, 0.08547608, 0.11414898, 0.14269586, 0.17132208 ,0.19979544, 0.22812643] 
twenty_c_median_3 = [ 29.9,  42.4,  57.1,  73.2,  89.1, 103.3, 118.8, 133.2]
twenty_c_height_3 = [0.02862272, 0.05643559, 0.08494162, 0.11359585, 0.14208555, 0.1705624, 0.19900192, 0.22757213] 
twenty_c_median_4 = [ 33.3,  37.8,  53.3,  70.3,  87.2, 104.9, 120.,  134.7]
twenty_c_height_4 = [0.02858188, 0.05708091, 0.08568846, 0.11432286, 0.14271803, 0.17118788, 0.20001366, 0.22849868] 

plt.plot(normalize_to_minus1_1(twenty_c_median_1), twenty_c_height_1, label='20° C - Trial 1', color='orange')
plt.plot(normalize_to_minus1_1(twenty_c_median_2), twenty_c_height_2, label='20° C - Trial 2', color='orange')
plt.plot(normalize_to_minus1_1(twenty_c_median_3), twenty_c_height_3, label='20° C - Trial 3', color='orange')
plt.plot(normalize_to_minus1_1(twenty_c_median_4), twenty_c_height_4, label='20° C - Trial 4', color='orange')

"""30° C"""
# average_internal_temperature = [29.95, 29.5, 29.2, 28.65, 28.75, 28.9, 29.05, 29.35]
# average_water_temperature = [31.5, 31.15, 31.05, 30.8, 30.6, 30.35, 29.9, 29.2]


thirty_c_median_1 = [136.8, 125.1, 140.3, 151.1, 164.5, 172.6, 180.9, 190.6]
thirty_c_heights_1 = [0.02850863, 0.05679118, 0.08509354, 0.11344951, 0.14178567, 0.17040617, 0.19885653, 0.22757375]
thirty_c_median_2 = [-26.1,  78.2, 103.9, 121.3, 139.3, 153.2, 164.3, 175.7]
thirty_c_heights_2 = [0.02861584, 0.05712097, 0.08526019, 0.11405665, 0.14249885, 0.17073013, 0.19948464, 0.22765649]

plt.plot(normalize_to_minus1_1(thirty_c_median_1), thirty_c_heights_1, label='30° C - Trial 1', color='red')
plt.plot(normalize_to_minus1_1(thirty_c_median_2), thirty_c_heights_2, label='30° C - Trial 2', color='red')

plt.xlabel("Strain (µε)", fontsize=12)
plt.ylabel("Height (m)", fontsize=12)
plt.title("Compensated Callibration Curves for Diaphragm FBG's at 0°C, 10°C, 20°C and 30°C", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

