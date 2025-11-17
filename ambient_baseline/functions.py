import pandas as pd
import numpy as np
import math

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

def find_true_height(mass, rho, r):
    height = (mass/1000)/(rho*math.pi*math.pow(r/2,2))
    return height