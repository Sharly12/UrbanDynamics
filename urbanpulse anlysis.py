import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
from matplotlib.gridspec import GridSpec
import os

# === 1. Load Data and Metadata ===
file_path = r"D:\University Repository\Academics\Level 4\Semester_8\Research\Urban_Pulse_Final_Findings\Analysis for 50 Cities\Urban_Pulse_Gravity_Cluster\Sapce_Time_Cube\Weekday_STC_only\Colombo_zones_outputs\Colombo_zones_space_time_cube.csv"
location_name = os.path.basename(file_path).split('_')[0]

df = pd.read_csv(file_path)
df['time_block'] = pd.to_datetime(df['time_range'].str.extract(r'^(\d{2}:\d{2})')[0], format='%H:%M')
df['minutes'] = df['time_block'].dt.hour * 60 + df['time_block'].dt.minute

# === 2. Aggregate and Smooth ===
pulse_15min = df.groupby('minutes')['count'].sum().reset_index()
x = pulse_15min['minutes']
y = pulse_15min['count']

x_smooth = np.linspace(x.min(), x.max(), 300)
spl = make_interp_spline(x, y, k=3)
y_smooth = spl(x_smooth)

hourly_activity = df.groupby(df['time_block'].dt.hour)['count'].sum()
x_hour = hourly_activity.index * 60
y_hour = hourly_activity.values
x_hour_smooth = np.linspace(x_hour.min(), x_hour.max(), 300)
spl_hour = make_interp_spline(x_hour, y_hour, k=3)
y_hour_smooth = spl_hour(x_hour_smooth)

# === 3. Change Point Detection ===
dy = np.gradient(y_smooth, x_smooth)
threshold = np.percentile(np.abs(dy), 90)
change_indices = np.where(np.abs(dy) > threshold)[0]
change_points = [change_indices[0]]
for idx in change_indices:
    if idx - change_points[-1] > 15:
        change_points.append(idx)
change_times = x_smooth[change_points]

# === 4. Segment Stats ===
segment_boundaries = [x_smooth[0]] + list(change_times) + [x_smooth[-1]]
segment_info = []
total_signal = np.sum(y_smooth)

for i in range(len(segment_boundaries) - 1):
    start = segment_boundaries[i]
    end = segment_boundaries[i + 1]
    mask = (x_smooth >= start) & (x_smooth < end)
    segment_y = y_smooth[mask]
    if len(segment_y) == 0: continue
    segment_info.append({
        "Segment": f"Segment {i+1}",
        "Start Time": pd.to_datetime(start, unit='m').strftime('%H:%M'),
        "End Time": pd.to_datetime(end, unit='m').strftime('%H:%M'),
        "Duration (min)": round(end - start),
        "Total Activity": round(np.sum(segment_y)),
        "Mean Activity": round(np.mean(segment_y), 2),
        "Std Dev": round(np.std(segment_y), 2),
        "% of Total": round(np.sum(segment_y) / total_signal * 100, 2)
    })

segment_df = pd.DataFrame(segment_info).dropna().reset_index(drop=True)

# === 5. General Metrics ===
total_activity = df['count'].sum()
std_dev = df.groupby('time_block')['count'].sum().std()
peak_time = pd.to_datetime(pulse_15min.iloc[y.argmax()]['minutes'], unit='m').strftime('%H:%M')
peaks, _ = find_peaks(y_smooth, prominence=np.percentile(y_smooth, 80))
valleys, _ = find_peaks(-y_smooth)
morning_activity = df[df['time_block'].dt.hour < 12]['count'].sum()
evening_activity = df[df['time_block'].dt.hour >= 12]['count'].sum()
am_pm_ratio = round(morning_activity / evening_activity, 2) if evening_activity != 0 else np.nan
am_pm_skew = "Morning" if morning_activity > evening_activity else "Evening"

# === 6. Advanced Metrics ===
cv = round(np.std(y_smooth) / np.mean(y_smooth), 3)
max_rise = round(np.max(dy), 2)
max_fall = round(np.min(dy), 2)
num_rebounds = len(peaks) + len(valleys)

# Peak hour duration
peak_threshold = 0.9 * max(y_smooth)
sustained_peak_minutes = np.sum(y_smooth >= peak_threshold)
peak_hour_duration = round(sustained_peak_minutes * (1440 / len(y_smooth)))

# === 7. Segment Colors ===
norm_activity = (segment_df["% of Total"] - segment_df["% of Total"].min()) / \
                (segment_df["% of Total"].max() - segment_df["% of Total"].min() + 1e-6)
colors = plt.cm.viridis(norm_activity.values)

# === 8. Plot Full Report ===
plt.figure(figsize=(16, 9))
gs = GridSpec(3, 2)

# 8.1 Main Pulse Curve
ax0 = plt.subplot(gs[0, :])
ax0.plot(x_smooth, y_smooth, label='Smoothed Urban Pulse (15-min)', linewidth=2, color='orange')
ax0.plot(x_hour_smooth, y_hour_smooth, label='Smoothed Hourly Pulse', linewidth=2, linestyle='--', color='navy')
ax0.scatter(x_smooth[peaks], y_smooth[peaks], color='red', label='Peaks')
for cp in change_times:
    ax0.axvline(cp, color='blue', linestyle='--', alpha=0.4)
for i, row in segment_df.iterrows():
    start = pd.to_datetime(row['Start Time'], format='%H:%M')
    end = pd.to_datetime(row['End Time'], format='%H:%M')
    start_min = start.hour * 60 + start.minute
    end_min = end.hour * 60 + end.minute
    ax0.axvspan(start_min, end_min, color=colors[i], alpha=0.35)
    mid_x = (start_min + end_min) / 2
    label = f"{row['Segment']}\n{row['Start Time']}–{row['End Time']}\n{row['% of Total']:.1f}%"
    ax0.text(mid_x, max(y_smooth)*0.95, label, ha='center', va='top', fontsize=9, rotation=90)

ax0.set_title(f"Segmented Urban Pulse Curve – {location_name}")
ax0.set_xlabel("Time of Day")
ax0.set_ylabel("Mobile Activity Count")
ax0.set_xticks(np.linspace(0, 1440, 13))
ax0.set_xticklabels(pd.to_datetime(np.linspace(0, 1440, 13), unit='m').strftime('%H:%M'), rotation=45)
ax0.legend(loc='upper left', fontsize='small')
ax0.grid(True)

# 8.2 AM/PM Colored Bar
ax1 = plt.subplot(gs[1, 0])
bar_colors = ['#a1d99b' if hr < 12 else '#fc9272' for hr in hourly_activity.index]
ax1.bar(hourly_activity.index, hourly_activity.values, color=bar_colors)
ax1.set_title("Hourly Activity – AM/PM Colored")
ax1.set_xlabel("Hour")
ax1.set_ylabel("Total Mobile Activity")
ax1.set_xticks(range(0, 24))
ax1.axvspan(0, 11.5, color='#a1d99b', alpha=0.15, label='AM')
ax1.axvspan(11.5, 23.5, color='#fc9272', alpha=0.15, label='PM')
ax1.legend()
ax1.grid(True)

# 8.3 AM/PM Pie
ax2 = plt.subplot(gs[1, 1])
ax2.pie([morning_activity, evening_activity], labels=['AM', 'PM'], autopct='%1.1f%%', colors=['#a1d99b', '#fc9272'])
ax2.set_title("AM vs PM Activity Share")

# 8.4 Text Summary
ax3 = plt.subplot(gs[2, :])
ax3.axis('off')
summary_text = f"""
📊 Urban Pulse Summary – {location_name}

• Total Mobile Activity: {total_activity:,}
• Peak Time of Day: {peak_time}
• Number of Distinct Peaks: {len(peaks)}
• Number of Detected Change Points: {len(change_times)}
• Activity Std Dev: {std_dev:.2f}
• Coefficient of Variation: {cv}
• Max Rise Rate: {max_rise}
• Max Fall Rate: {max_fall}
• Peak Hour Duration (≥90%): {peak_hour_duration} min
• Rebounds (Peaks + Valleys): {num_rebounds}
• AM/PM Ratio: {am_pm_ratio}
• Temporal Skewness: Higher in the {am_pm_skew}
• Total Segments Identified: {len(segment_df)}
"""
ax3.text(0, 0.95, summary_text, fontsize=11, va='top', fontfamily='monospace')

plt.tight_layout()
plt.show()

# === 9. Save segment stats ===
segment_df.to_csv(f"{location_name}_segment_statistics.csv", index=False)
