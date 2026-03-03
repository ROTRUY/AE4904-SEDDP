### IMPORTS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import defaultdict
from CONSTANTS import dataVolume, altitude
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

### GROUND STATION NETWORKS
# Format: "StationName": (Latitude [deg], Longitude [deg], Altitude [km], Minimum Elevation [deg])
# Optical
OpticalNetwork = {
                "Delft":    (51.99,   4.38, 0.060, 30), 
                "Granada":  (37.00,  -3.20, 0.738, 30), 
                "Tenerife": (28.30, -16.51, 2.400, 30), 
                "Nemea":    (37.85,  22.62, 0.300, 30), 
                "Nicosia":  (34.80,  33.40, 0.220, 30), 
                "Porto":    (41.50,  -8.80, 0.100, 30)
                }

RFFromOptical = {
                "Delft":    (51.99,   4.38, 0.060, 10), 
                "Granada":  (37.00,  -3.20, 0.738, 10), 
                "Tenerife": (28.30, -16.51, 2.400, 10), 
                "Nemea":    (37.85,  22.62, 0.300, 10), 
                "Nicosia":  (34.80,  33.40, 0.220, 10), 
                "Porto":    (41.50,  -8.80, 0.100, 10)
                }

# "Weilheim":  (47.88, 11.08, 0.563, 10), 

RFNetwork = {
                "Redu":       (50.00,  5.15,  0.387, 10), 
                "Cebreros":   (40.45, -4.37,  0.794, 10), 
                "Maspalomas": (27.76, -15.63, 0.205, 10),
                "Fucino":     (41.98,  13.60, 0.661, 10)
                }

SSONetwork = {
                "Svalbard": (78.22, 15.63, 0, 10),
                "Trollsat": (-72.00,  2.53, 0, 10),
                }


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
        # === Initialise variables ===
        self.data: dict[str, list[(float, float, float)]] = defaultdict(list) # {station: [(start, stop, duration), ...]}
        self.start: datetime  # Start epoch of time interval
        self.stop: datetime  # Stop epoch of time interval
        self.length: int  # Length of time interval in days
        self.contactPerDay: dict[datetime, timedelta] = defaultdict(timedelta)  # Contact time per day {date, time}
        self.totalContactTime: timedelta = timedelta()  # Total contact time
        self.avgContactTime: float  # Average contact time per day [s]
        self.stations = stations  # The stations to take into account, if empty it will account for them all.

        # === Read file ===
        with open(f"GMATContacts\{filename}") as f:
            for line in f:
                if line[:10] == "Observer: ":
                    station = line[10:-1]
                if line[0] in ("0", "1", "2", "3"):
                    start = datetime.strptime(line[:24], "%d %b %Y %H:%M:%S.%f")
                    stop = datetime.strptime(line[28:52], "%d %b %Y %H:%M:%S.%f")
                    duration = float(line[58:70])
                    self.data[station].append((start, stop, duration))
        
        # === Calculate everything ===
        self.contactTime()
    
            
    def plot(self, show: bool = True, save: bool = False, name: str = "ContactPlot") -> None:
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
    

    def plotMap(self, network: dict[str, tuple[float, float, float, float]], altitude: float, inclination: float, groundtrack_file: str, show: bool = True, save: bool = False, name: str = "ContactMap") -> None:
        """
        Plots the ground stations and their visibility footprints on a world map.
        :param network: Dictionary of station coordinates and parameters. Format: {station_name: (latitude [deg], longitude [deg], altitude [km], min elevation [deg])}
        :type network: dict[str, tuple[float, float, float, float]]
        :param altitude: Altitude of the satellite. [km]
        :type altitude: float
        :param show: Whether you want the method to show the plot when ran.
        :type show: bool
        :param save: Whether you want the method to save the plot when ran.
        :type save: bool
        :param name: Name for the plot when saved.
        :type name: str
        """

        R_E = 6371  # km
        r_s = R_E + altitude

        # === Create map with projection ===
        fig = plt.figure(figsize=(14, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # === Add Earth features ===
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_global()

        ax.set_title(f"Ground Stations and Visibility Footprints (h = {altitude} km, i = {inclination} deg)")

        # === Plot Ground Track ===
        if groundtrack_file is not None:
            df = pd.read_csv(groundtrack_file)

            lats = df.iloc[:, 1].values
            lons = df.iloc[:, 2].values

            ax.scatter(lons,
                    lats,
                    s=1, # point size
                    color='gray',
                    alpha=0.1,
                    transform=ccrs.PlateCarree())

        # === Loop through stations ===
        for station, (lat_deg, lon_deg, h_g, min_elev_deg) in network.items():

            r_g = R_E + h_g
            E = np.radians(min_elev_deg)

            # === Compute Earth central angle ψ ===
            term1 = (r_g / r_s) * np.cos(E)**2
            term2 = np.sin(E) * np.sqrt(1 - ((r_g / r_s) * np.cos(E))**2)
            cos_psi = np.clip(term1 + term2, -1, 1)
            psi = np.arccos(cos_psi)

            # === Plot station ===
            ax.plot(lon_deg, lat_deg,
                    marker='o',
                    color='red',
                    markersize=5,
                    transform=ccrs.PlateCarree())

            ax.text(lon_deg + 2, lat_deg + 2,
                    station,
                    fontweight='bold',
                    transform=ccrs.PlateCarree())

            # === Generate footprint circle ===
            circle_lats = []
            circle_lons = []

            lat0 = np.radians(lat_deg)
            lon0 = np.radians(lon_deg)

            for theta in np.linspace(0, 2*np.pi, 300):

                lat = np.arcsin(
                    np.sin(lat0)*np.cos(psi) +
                    np.cos(lat0)*np.sin(psi)*np.cos(theta)
                )

                lon = lon0 + np.arctan2(
                    np.sin(theta)*np.sin(psi)*np.cos(lat0),
                    np.cos(psi) - np.sin(lat0)*np.sin(lat)
                )

                circle_lats.append(np.degrees(lat))
                circle_lons.append(np.degrees(lon))

            ax.plot(circle_lons,
                    circle_lats,
                    transform=ccrs.PlateCarree(),
                    alpha=0.8)

        if save:
            plt.savefig(f"Plots\\{name}.png", dpi=300, bbox_inches="tight")

        if show:
            plt.show()
    
    
    def contactTime(self) -> None:
        """
        Calculates all necessary factors to evaluate contact with ground station.
        """
        intervals = list()
        merged = list()

        for station in sorted(self.data.keys()):
            # Only count the times if the station is in the input list, or if no input list was given.
            if self.stations is None or station in self.stations:
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
    
    def summary(self, Print:bool=True, save:bool=False, name:str = "ContactSummary") -> None:
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
            f = open(f"ContactSummaries\{name}.txt", "w")
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

def availability(P_contact: list[float]) -> float:
    """
    Estimates availability based on list of cloud-free line-of-sight probabilities for optical, or for list of rain probabilities for RF.
    Assumes probabilities are independent.

    :param P_contact: List of contact probabilities.
    :type P_contact: list[float]
    """
    P_contact = np.array(P_contact)
    P_outage = np.prod(1 - P_contact)
    return 1 - P_outage

### RUN HERE
# Optical Network: ["Delft", "Granada", "Tenerife", "Nemea", "Nicosia", "Porto"]
#print(availability([.35, .6708, .60, .60, .7931, .5962])) # Optical network availability: 0.997139644775104

# For RF, instead of P_CFLOS we use percentage of rainy days in the year
# RF Network: ["Redu", "Cebreros", "Maspalomas", "Fucino"]
# print(availability([.5069, .8767, .9288, .7260])) # RF network availability: 0.998813879981776

inclination = 98

opt_groundtrack = "GMATReports\DelftReport.csv"
rf_groundtrack = "GMATReports\CebrerosReport.csv"
sso_groundtrack = "GMATReports\SvalbardReport.csv"

"""
opticalTimes = ContactTimes(f"Opt{inclination}.txt")
#opticalTimes.summary(False, True, f"OptSum{inclination}")
#opticalTimes.plot(False, True, f"OptPlot{inclination}")
opticalTimes.plotMap(OpticalNetwork, 550, inclination, None, False, True, f"OptMap")

rfTimes = ContactTimes(f"RF{inclination}.txt")
#rfTimes.summary(False, True, f"RFSum{inclination}")
#rfTimes.plot(False, True, f"RFPlot{inclination}")
rfTimes.plotMap(RFNetwork, 550, inclination, None, False, True, f"RFMap")
"""

ssoTimes = ContactTimes(f"SSO.txt")
ssoTimes.summary(True, False, f"SSOSum")
ssoTimes.plotMap(SSONetwork, 550, inclination, sso_groundtrack, True, False, f"SSOMap")