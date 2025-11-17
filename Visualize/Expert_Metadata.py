import pandas as pd
import matplotlib.pyplot as plt

path = "C:\junha\Datasets\LTDv2\metadata_images.csv"

df = pd.read_csv(path)
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["time_hour"] = (
    df["DateTime"].dt.hour
    + df["DateTime"].dt.minute / 60
    + df["DateTime"].dt.second / 3600
)
df["T_minus_D"] = df["Temperature"] - df["Dew Point"]

for col in [
    "Sun Radiation Intensity",
    "Humidity",
    "Precipitation",
    "T_minus_D",
]:
    v = df[col].astype(float).values
    df[col + "_z"] = (v - v.mean()) / (v.std() + 1e-8)

R = df["Sun Radiation Intensity_z"]
H = df["Humidity_z"]
P = df["Precipitation_z"]
D = df["T_minus_D_z"]

df["V_raw"] = 0.7 * R + 0.4 * D - 0.7 * H - 0.6 * P


plt.hist(df["V_raw"], bins=50)
plt.xlabel("V_raw")
plt.ylabel("count")
plt.show()
