### IMPORTS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import defaultdict
from CONSTANTS import dataVolume
from math import prod
import numpy as np

### CLASSES
class ContactTimes():
    """
    Class for calculating stuff about the contact times.
    """

    def __init__(self, filename: str, stations: list[str]=None) -> None:
        """
        Initialiser.

        :param filename: Name of the text file with the contactLocator data from GMAT.
        :type filename: str
        :param stations: List of station names (strings) to take into account. If none are given, all are taken into account. Names must be accurate!
        :type stations: list[str]
        """
        # Initialise variables
        self.data: dict[str, list[(float, float, float)]] = defaultdict(list) # {station: [(start, stop, duration), ...]}
        self.start: datetime  # Start epoch of time interval
        self.stop: datetime  # Stop epoch of time interval
        self.length: int  # Length of time interval in days
        self.contactPerDay: dict[datetime, timedelta] = defaultdict(timedelta)  # Contact time per day {date, time}
        self.totalContactTime: timedelta = timedelta()  # Total contact time
        self.avgContactTime: float  # Average contact time per day [s]
        self.stations = stations  # The stations to take into account, if empty it will account for them all.

        # Read file
        with open(f"GMATContacts\{filename}") as f:
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
            plt.savefig(f"Plots\{name}.png")
    
    
    def contactTime(self) -> None:
        """
        Calculates all necessary factors to evaluate contact with ground station.
        """
        intervals = list()
        merged = list()

        for station in sorted(self.data.keys()):
            # Only count the times if the station is in the input list, or if no input list was given.
            if not self.stations or station in self.stations:
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
    
    def summary(self, Print:bool=True, save:bool=False, name:str = f"ContactSummary {datetime.now()}") -> None:
        """
        Prints summary of calculated factors.

        :param print: Whether or not to print the summary. (Default True)
        :type print: bool
        :param save: Whether or not to save the summary to a text file. (Default False)
        :type save: bool
        :param name: Name to give the summary file if saved.
        :type name: str
        """
        if Print:
            print("\n========== CONTACT SUMMARY ==========\n")

            print("Stations taken into account:\n")
            if self.stations:
                print(f"{self.stations}\n")
            else:
                print("All\n")

            print(f"\nAnalysis window:")
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

        if save:
            f = open(f"ContactSummaries\{name}.txt", "x")
            with f:
                f.write("========== CONTACT SUMMARY ==========\n")

                f.write("Stations taken into account:\n")
                if self.stations:
                    f.write(f"{self.stations}\n")
                else:
                    f.write("All\n")

                f.write(f"\nAnalysis window:\n")
                f.write(f"Start: {self.start}\n")
                f.write(f"End:   {self.stop}\n")
                f.write(f"Duration: {self.length} days\n")

                f.write(f"\nTotal contact time:\n")
                f.write(f"{self.totalContactTime} = {self.totalContactTime.total_seconds() / 60} minutes\n")
                
                f.write("\nContact time per day:\n")
                for day in sorted(self.contactPerDay.keys()):
                    minutes = self.contactPerDay[day].total_seconds() / 60
                    f.write(f"{day} : {minutes:.3f} min\n")

                f.write("\nAverage contact per day: \n")
                f.write(f"{self.avgContactTime / 60:.3f} min/day\n")

                f.write(f"\nData rate required based on average: {dataVolume / self.avgContactTime * 1e6:.3f} Mbps\n")

def availability(P_CFLOS: list[float]) -> float:
    """
    Estimates availability based on list of cloud-free line-of-sight probabilities.
    Assumes probabilities are independent.

    :param P_CFLOS: List of cloud-free line-of-sight probabilities.
    :type P_CFLOS: list[float]
    """
    P_CFLOS = np.array(P_CFLOS)
    P_outage = np.prod(1 - P_CFLOS)
    return 1 - P_outage

### RUN HERE
# sixA: ["Delft", "Granada", "Tenerife", "Nemea", "Nicosia", "Porto"]
#print(availability([.35, .6708, .60, .60, .7931, .5962]))  # 0.997139644775104

times = ContactTimes("sixA50RF.txt")
times.summary(False, True, "sixA50RF")
times.plot(False, True, "sixA50RFPlot")

