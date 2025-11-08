import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r"C:\junha\Datasets\LTDv2\metadata_images.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["Date"] = df["DateTime"].dt.date
df["Hour"] = df["DateTime"].dt.hour
df.columns = df.columns.str.strip().str.replace(" ", "_")

print(df.describe())

daily = df.groupby("Date")[["Temperature", "Humidity", "Sun_Radiation_Intensity", "Wind_Speed"]].mean()
daily.plot(subplots=True, figsize=(10,8), title=["Temperature (°C)", "Humidity (%)", "Sun Radiation (W/m²)", "Wind Speed (m/s)"])
plt.tight_layout()
plt.show()

hourly = df.groupby("Hour")[["Temperature", "Sun_Radiation_Intensity", "Humidity"]].mean()
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(hourly.index, hourly["Temperature"], label="Temperature (°C)")
ax.plot(hourly.index, hourly["Sun_Radiation_Intensity"], label="Sun Radiation (W/m²)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Value")
ax.set_title("Hourly Average (Temperature & Solar Radiation)")
ax.legend()
plt.show()

weather_cols = ["Temperature", "Humidity", "Precipitation", "Dew_Point", "Wind_Speed", "Sun_Radiation_Intensity"]
corr = df[weather_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Weather Variables")
plt.show()

sns.pairplot(df[weather_cols], corner=True)
plt.suptitle("Pairwise Relationship between Weather Variables", y=1.02)
plt.show()

bins = [0, 100, 500, 800, 1200]
labels = ["매우 흐림", "흐림", "맑음", "매우 맑음"]
df["SunCategory"] = pd.cut(df["Sun_Radiation_Intensity"], bins=bins, labels=labels)
plt.figure(figsize=(8,4))
sns.countplot(x="SunCategory", data=df, order=labels)
plt.title("Distribution by Solar Radiation Category")
plt.xlabel("Sun Radiation Category")
plt.ylabel("Frame Count")
plt.show()
