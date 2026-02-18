### IMPORTS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import defaultdict
from CONSTANTS import dataVolume

### CLASSES
class ContactTimes():
    """
    Class for calculating stuff about the contact times.

    :param filename: Name of the text file with the contactLocator data from GMAT (don't include .txt).
    :type filename: str
    """

    def __init__(self, filename: str) -> None:
        """
        Initialiser.

        :param filename: Name of the text file with the contactLocator data from GMAT (don't include .txt).
        :type filename: str
        """
        # Initialise variables
        self.data: dict[str, list[(float, float, float)]] = defaultdict(list) # {station: [(start, stop, duration), ...]}
        self.start: datetime  # Start epoch of time interval
        self.stop: datetime  # Stop epoch of time interval
        self.length: int  # Length of time interval in days
        self.contactPerDay: dict[datetime, timedelta] = defaultdict(timedelta)  # Contact time per day {date, time}
        self.totalContactTime: timedelta = timedelta()  # Total contact time
        self.avgContactTime: float  # Average contact time per day [s]

        # Read file
        with open(f"{filename}.txt") as f:
            for line in f:
                if line[:10] == "Observer: ":
                    station = line[10:-1]
                if line[0] in ("0", "1", "2", "3"):
                    start = datetime.strptime(line[:24], "%d %b %Y %H:%M:%S.%f")
                    stop = datetime.strptime(line[28:52], "%d %b %Y %H:%M:%S.%f")
                    duration = float(line[58:70])
                    self.data[station].append((start, stop, duration))
        
        # Calculate everything
        self.contactTime()
    
            
    def plot(self, show: bool = True, save: bool = False, name: str = f"ContactPlot {datetime.now()}") -> None:
        """
        Method to plot the contact times per ground station.
        
        :param show: Whether you want the method to show the plot when ran.
        :type show: bool
        :param save: Whether you want the method to save the plot when ran.
        :type save: bool
        :param name: Name for the plot when saved.
        :type name: str
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        station_names = sorted(self.data.keys())

        for i, station in enumerate(station_names):
            for start, stop, _ in self.data[station]:
                ax.barh(
                    y=i,
                    width=(stop - start).total_seconds() / 60 / 60 / 24,  # convert to days
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

        if show:
            plt.show()

        if save:
            plt.savefig(f"{name}.png")
    
    
    def contactTime(self) -> None:
        """
        Calculates all necessary factors to evaluate contact with ground station.
        """
        intervals = list()
        merged = list()

        for station in sorted(self.data.keys()):
            for start, stop, _ in self.data[station]:
                intervals.append((start, stop))

        intervals.sort(key=lambda x: x[0])

        for start, stop in intervals:
            if not merged:
                merged.append([start, stop])
            else:
                last_start, last_stop = merged[-1]
                if start <= last_stop:
                    merged[-1][1] = max(last_stop, stop)
                else:
                    merged.append([start, stop])

        for start, stop in merged:
            self.totalContactTime += (stop - start)

        for start, stop in merged:
            current = start

            while current < stop:
                end_of_day = datetime.combine(current.date() + timedelta(days=1), datetime.min.time())

                segment_end = min(stop, end_of_day)
                self.contactPerDay[current.date()] += (segment_end - current)
                current = segment_end

        self.start = min(start for start, stop in merged)
        self.stop   = max(stop for start, stop in merged)
        self.length = (self.stop.date() - self.start.date()).days + 1
        self.avgContactTime = self.totalContactTime.total_seconds() / self.length  # Average contact time per day [s]
    
    def summary(self) -> None:
        """
        Prints summary of calculated factors.
        """
        print("\n========== CONTACT SUMMARY ==========\n")

        print(f"Analysis window:")
        print(f"Start: {self.start}")
        print(f"End:   {self.stop}")
        print(f"Duration: {self.length} days\n")

        print(f"Total contact time:")
        print(f"{self.totalContactTime} = {self.totalContactTime.total_seconds() / 60} minutes")

        print("Contact time per day:")
        for day in sorted(self.contactPerDay.keys()):
            minutes = self.contactPerDay[day].total_seconds() / 60
            print(f"{day} : {minutes:.3f} min")

        print("Average contact per day: "
            f"{self.avgContactTime / 60:.3f} min/day\n")

        print(f"Data rate required based on average: {dataVolume / self.avgContactTime * 1e6:.3f} Mbps\n")

### RUN HERE
# SYS2+ is entirity of SYS2 plus Tenerife
SYS2E = ContactTimes("ContactLocatorSYS2+")

# minimal99 is the minimum amount of the SYS2 ground stations to get 99% availability: Nicosia, Gibraltar, Barcelona, Lausanne, Naples and Porto
minimal99 = ContactTimes("minimal99")

# minimal99A is minimal99 plus Tenerife
minimal99A = ContactTimes("minimal99A")

# minimal99B is minimal99A plus Nemea
minimal99B = ContactTimes("minimal99B")

# minimal99C is minimal99B plus Delft
minimal99C = ContactTimes("minimal99C")

# Full is every single European optical ground station found
full = ContactTimes("full")
full.summary()