### IMPORTS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import defaultdict

### READ AND CONVERT DATA
# Initialise data
data = [] # ["station", "start", "stop", "duration"]

# Read file
with open("ContactLocatorSYS2+.txt") as f:
    for line in f:
        if line[:10] == "Observer: ":
            station = line[10:-1]
        if line[0] in ("0", "1", "2", "3"):
            # Example start/stop: 13 Feb 2026 03:46:42.958 ==> %d %b %Y %H:%M:%S.%f
            start = datetime.strptime(line[:24], "%d %b %Y %H:%M:%S.%f")
            stop = datetime.strptime(line[28:52], "%d %b %Y %H:%M:%S.%f")
            duration = float(line[58:70])
            data.append([station, start, stop, duration])

# Group contact times per station
stations = defaultdict(list)

for station, start, stop, duration in data:
    stations[station].append((start, stop))

### PLOT
SAVEFIG = True

fig, ax = plt.subplots(figsize=(14, 8))
station_names = sorted(stations.keys())

for i, station in enumerate(station_names):
    for start, stop in stations[station]:
        ax.barh(
            y=i,
            width=(stop - start).total_seconds() / 86400,  # convert to days
            left=mdates.date2num(start),
            height=0.6
        )

ax.set_yticks(range(len(station_names)))
ax.set_yticklabels(station_names)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xticks(rotation=45)
ax.set_xlabel("Time")
ax.set_ylabel("Ground Station")
ax.set_title("Ground Station Contact Windows")

plt.tight_layout()
if SAVEFIG:
    plt.savefig("ContactTimesPlotSYS2+.png")

### CALCULATE STUFF
# Actual contact time, taking into account overlaps
intervals = [(start, stop) for station, start, stop, duration in data]
intervals.sort(key=lambda x: x[0])

merged = []

for start, stop in intervals:
    if not merged:
        merged.append([start, stop])
    else:
        last_start, last_stop = merged[-1]
        if start <= last_stop:
            merged[-1][1] = max(last_stop, stop)
        else:
            merged.append([start, stop])

total_contact = timedelta()

for start, stop in merged:
    total_contact += (stop - start)

contact_per_day = defaultdict(timedelta)

for start, stop in merged:
    current = start

    while current < stop:
        end_of_day = datetime.combine(
            current.date() + timedelta(days=1),
            datetime.min.time()
        )

        segment_end = min(stop, end_of_day)
        contact_per_day[current.date()] += (segment_end - current)
        current = segment_end

start_time = min(start for start, stop in merged)
end_time   = max(stop for start, stop in merged)
calendar_days = (end_time.date() - start_time.date()).days + 1
average_minutes = total_contact.total_seconds() / 60 / calendar_days

print("\n========== CONTACT SUMMARY ==========\n")

print(f"Analysis window:")
print(f"Start: {start_time}")
print(f"End:   {end_time}")
print(f"Duration: {calendar_days:.3f} days\n")

print(f"Total contact time:")
print(f"{total_contact} = {total_contact.total_seconds() / 60} minutes")

print("Contact time per day:")
for day in sorted(contact_per_day.keys()):
    minutes = contact_per_day[day].total_seconds() / 60
    print(f"{day} : {minutes:.3f} min")

print("Average contact per day: "
      f"{average_minutes:.3f} min/day\n")