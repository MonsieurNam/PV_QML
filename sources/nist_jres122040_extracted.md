Volume 122, Article No. 40 (2017) https://doi.org/10.6028/jres.122.040
Journal of Research of the National Institute of Standards and Technology
Performance Data from the NIST Photovoltaic
Arrays and Weather Station
Matthew Boyd
National Institute of Standards and Technology,
Gaithersburg, MD 20899, USA
matthew.boyd@nist.gov
Data DOI: https://doi.org/10.18434/M3S67G
Key words: data acquisition; inverter; meteorology; photovoltaic; PV; solar; weather station.
Accepted: October 27, 2017
Published: November 1, 2017
https://doi.org/10.6028/jres.122.040
1. Summary
In July 2012, the National Institute of Standards and Technology (NIST) completed construction of three
photovoltaic (PV) arrays on its Gaithersburg, MD campus. Comprehensive data acquisition systems were installed
and an onsite weather station was also built to collect ancillary solar and meteorological measurements that are
needed for the full characterization and modeling of the PV arrays. These datasets provide high-resolution, low-
uncertainty, comprehensive PV performance and weather data for extended, continuous time periods. The creation
of these datasets is fulfilling a need of the research and energy communities that few other datasets meet. Data from
these systems have been collected for about three years at the time of this publication, between August 2014 and
July 2017, and are being provided to the public via an online web portal for viewing and download.
The three arrays on campus use the same monocrystalline silicon module but are mounted in different
configurations: the Canopy Array is mounted on east/west facing canopies over a parking lot, the Ground Array is
mounted on tilted ground supports in an open field, and the Roof Array is mounted on tilted, weighted racks on a flat
building roof. There is a single inverter at each array that is connected to the local grid via the NIST campus grid.
The arrays are instrumented with research-grade sensors measuring various electrical, temperature, and
meteorological parameters. At each array there are wind, ambient temperature, and various solar measurements
being conducted. There is also a comprehensive weather station installed on a nearby building. The weather station
includes instruments measuring the various solar components, spectral curves, ultraviolet (UV) light and infrared
radiation (IR) energy, and a full suite of standard weather conditions. Reference modules are also installed near the
weather station in the same manner and orientations as those in the arrays. These modules are maximum power
tracked, with the current-voltage (I-V) curves traced every minute. Cameras capture images of these reference
modules, the arrays, and the entire sky.
2. Data Specifications
Nearly all of the 360+ measurements at the arrays and weather station are sampled and saved every second,
with 1-minute averages also saved. Exceptions are the all-sky images, which are saved every 5 seconds to 8 seconds,
the module backsheet temperatures sampled every 10 seconds (but saved every 1 second), the spectroradiometer
irradiance curves sampled and saved every 1 minute, and the images of the arrays saved every 5 minutes. NIST is
making available 1-minute average values for the measurements and 1-hour intervals for the images during the first
1
How to cite this article:
Boyd M (2017) Performance Data from the NIST Photovoltaic
Arrays and Weather Station. J Res Natl Inst Stan 122:40.
https://doi.org/10.6028/jres.122.040

Volume 122, Article No. 40 (2017) https://doi.org/10.6028/jres.122.040
Journal of Research of the National Institute of Standards and Technology
two complete calendar years of testing (2015 and 2016) via a web portal1 [1]; the 1-second instantaneous, higher
frequency images, and data outside this time period are available upon request. Additional specifications are
provided below.
Table 1. Basic dataset metadata.
NIST Operating Unit Engineering Laboratory; Energy and Environment Division
Format Text (Comma Separated Values, XML), Images (JPEG)
Instrument meteorological, solar, electrical, temperature, images
Spatial or Temporal Elements Spatial (Latitude [°N], Longitude [°E], Elevation [m]):
Canopy array = (39.1385°, -77.2155°, 137 m)
Ground array = (39.1319°, -77.2141°, 138 m)
Roof array = (39.1354°, -77.2156°, 149 m)
Weather station = (39.1374°, -77.2187°, 158 m)
Temporal: 2015-01-01 through 2016-12-31
Data Dictionary https://www.nist.gov/file/391591 [pdf]
Accessibility All datasets submitted to Journal of Research of NIST are publicly available.
License https://www.nist.gov/director/licensing
3. Methods
Data were sampled and saved on a 1-second basis for nearly all of the measurements, with the main exception
being the resistance temperature detectors measuring the module backsheet temperature, which were sampled at a
10-second rate. All measurements are aggregated to 1-minute values, using either an average, minimum, maximum,
or sum, whichever is most appropriate. The reference module maximum power voltage and current were sampled
every second, but only the 1-minute aggregate values were saved. I-V curves of these reference modules were
performed and saved every minute. Images of the whole sky were saved every 8 seconds, and images of the arrays
were saved every 5 minutes. The timestamps associated with all measurements and images are in Eastern Standard
Time (no daylight saving time), and are within 1-second of the actual time as all data loggers, cameras, and
computers are synchronized to an internet time server every day just after midnight.
While every effort was made to minimize disruptions in data flow, there were times during the test phase when
data were not available or particular measurements were erroneous. A log of such instances is maintained on the
website where the data can be accessed. Data loss and errors were, however, kept to an extreme minimum, with
these two years of data having a combined data availability greater than 99 %. All radiometers at the PV arrays and
weather station are cleaned three times per week, with another log also kept of the precise times.
The reader is referred to the resources listed in Table 2 for details on the construction and operation of the test
facilities.
Table 2. Resources for further information on the PV arrays and weather station.
Website at https://www.nist.gov/el/energy-and-environment-
General Information: division-73200/heat-transfer-alternative-energy-
systems/photovoltaic-0
Design and construction specifications: Dougherty and Boyd 2017 [2]
Measurement approaches: Boyd 2015 [3] Boyd 2016 [4]
High-level performance modeling results: Boyd 2017 [5]
1 https://www.nist.gov/el/energy-and-environment-division-73200/heat-transfer-alternative-energy-systems/photovoltaic-0#data
2 https://doi.org/10.6028/jres.122.040

  Volume 122, Article No. 40 (2017) https://doi.org/10.6028/jres.122.040
  Journal of Research of the National Institute of Standards and Technology

The data website contains a data dictionary that includes two main sections, one for the arrays and one for the
weather station. Each table has the same format, which includes the base name of the measurement channel variable,
its units, a general description, the one-minute aggregate function, and an indicator of its location. The actual
variable names for the 1-second instantaneous values are the base names, but those variable names for the 1-minute
values are the base names appended with either an “_Avg”, “_Min”, “_Max”, or “_Sum” suffix per their aggregate
function. For the variable names ending in a number in parentheses, this suffix is positioned before the parentheses.
The location indicator, in the arrays section, indicates the array where this measurement was taken. For the weather
station, this indicator just indicates which of the two dataloggers at the weather station took the measurement.

4.  Configuration

The arrays were generally designed and installed to be most economical for the customer (NIST) and
installer/integrator, and their configurations have not changed since their construction. A summary of the arrays is
provided in Table 3.

Table 3. Summary of the NIST campus PV arrays.

|                            | Canopy    |     | Ground    | Roof      |
| -------------------------- | --------- | --- | --------- | --------- |
| Array Rated DC Power [kW]  | 243       |     | 271       | 73        |
| Latitude [°N]              | 39.1385   |     | 39.1319   | 39.1354   |
| Longitude [°E]             | -77.2155  |     | -77.2141  | -77.2156  |
| Elevationa [m]             | 137       |     | 138       | 149       |
| Heightb [m]                | 5.11      |     | 0.67      | 0.08      |
| Tilt [°]                   | 5         |     | 20        | 10        |
| Azimuth [° CW from N.]     | 90, 270   |     | 180       | 180       |
| Number of Modules          | 1032      |     | 1152      | 312       |
Sharp2
Module Manufacturer
| Module Model               |     |                                          | NU-U235F2  |     |
| -------------------------- | --- | ---------------------------------------- | ---------- | --- |
| Module Technology          |     | monocrystalline silicon – front contact  |            |     |
| Module Rated Power [W]     |     |                                          | 235        |     |
| Modules Per String         |     |                                          | 12         |     |
| Number of Source Circuits  | 86  |                                          | 96         | 26  |
| Number of Combiner Boxes   | 7   |                                          | 7          | 4   |
| Number of Inverters        | 1   |                                          | 1          | 1   |
Inverter Manufacturer  PV Powered (now Advanced Energy)  Satcon
| Inverter Model             |     | PVP260kW  |     | PVS-75 480  |
| -------------------------- | --- | --------- | --- | ----------- |
| Inverter Rated Power [kW]  |     | 260       |     | 75          |
a Elevation of the array support (e.g., ground, roof) above sea level.
b Height of bottom edge of the bottom module above the array support (e.g., ground, roof).

2 Certain commercial equipment, instruments, or materials are identified in this paper to foster understanding. Such identification does not imply
recommendation or endorsement by the National Institute of Standards and Technology, nor does it imply that the materials or equipment
identified are necessarily the best available for the purpose.

|     |     | 3   |     | https://doi.org/10.6028/jres.122.040   |
| --- | --- | --- | --- | -------------------------------------- |

Volume 122, Article No. 40 (2017) https://doi.org/10.6028/jres.122.040
Journal of Research of the National Institute of Standards and Technology
5. Disclaimers
The data provided are those measured from the instruments and sensors associated with the NIST PV arrays and
weather station. No efforts have been made to filter data or remove stray readings. Users should be aware that
occasional problems occur with any sensing system and should therefore examine data for any anomalies and assess
the validity of such data points. At times when no data are present from a particular instrument, an entry of “NAN”
is placed in the dataset. There may be other times when data are being recorded, but erroneous connections may
result in a series of missing values, or erroneous values that are inside or outside the physically possible domain.
Those values have not been removed, but the user is cautioned to be aware that they may exist in the dataset.
The experiment was monitored daily to ensure that the data acquisition and PV arrays operated normally, with
any problems resolved as quickly as possible. Any problems with the data acquisition systems or arrays has been
recorded in a document included with the dataset. Issues with the sensors, instruments, and data acquisition systems
are rare, and there are very few data gaps. However, the arrays have experienced failures and periods of maintenance
sometimes lasting several weeks. There were two significant events, both occurring at the Ground array: (1) a
complete removal and reinstallation of the modules, and (2) an electrical arcing event in the inverter. The first event
was to refurbish the prematurely rusting support structure, and lasted 33 days between 2015-11-02 and 2015-12-04.
The second event occurred on 2015-08-25, and was likely due to a loose fuse holder in the DC combiner
compartment. The inverter was brought back online on 2015-09-02, and destroyed shunt measuring instruments
were replaced on 2015-09-09.
6. Impact
It is anticipated that these datasets will be used by developers of PV modeling software for validation and
tuning, and by other PV modelers during the development of new module temperature, inverter, and whole-array
performance models. PV analysts will be able to better understand causes of module and balance-of-system (BOS)
component failures, dynamic electrical effects from fast moving clouds and general irradiance enhancement and
transitions, and the relative effect of different array orientations and configurations, to name a few examples. The
high-speed monitoring and sky imaging will allow smart-grid and solar forecasting researchers to evaluate and
validate short-term irradiance forecast and inverter-grid interaction models. The datasets also can be used as a
reference to test and standardize modeling approaches.
Acknowledgments
The research team would like to thank the following NIST Engineering Lab (EL) staff: Tzong Hao Chen for
creating the portals, Steve Barber for the storage solutions, and Steve Bannon for the web services and application
deployment. Much thanks are also given to EL’s Andrew Mundy for facilitating other related IT solutions.
7. References
[1] Photovoltaic Testbeds. Available at https://www.nist.gov/el/energy-and-environment-division-73200/heat-transfer-alternative-energy-
systems/photovoltaic-0. Accessed July 31, 2017.
[2] Dougherty B, Boyd M (2017) Model (At Least) Twice, Build Once: Experiences with the Design–Bid–Build Process for Solar Photovoltaic
Arrays. Journal of Solar Energy Engineering 139(3):035001. https://doi.org/10.1115/1.4036055.
[3] Boyd MT (2015) High-Speed Monitoring of Multiple Grid-Connected Photovoltaic Array Configurations. U.S. Department of Commerce,
Washington, D.C., NIST Technical Note 1896. https://doi.org/10.6028/nist.tn.1896.
[4] Boyd MT (2016) NIST Weather Station for Photovoltaic and Building System Research, National Institute of Standards and Technology.
U.S. Department of Commerce, Washington, D.C., NIST Technical Note 1913. https://doi.org/10.6028/nist.tn.1913.
[5] Boyd MT (2017) Comparative Performance and Model Agreement of Three Common Photovoltaic Array Configurations. Journal of Solar
Energy Engineering. https://doi.org/10.1115/1.4038314.
About the author: Matthew Boyd is the associate project leader of the photovoltaic project in the Heat Transfer and
Alternative Energy Systems Group in the Energy and Environment Division within the Engineering Laboratory. The
National Institute of Standards and Technology is an agency of the U.S. Department of Commerce.
4 https://doi.org/10.6028/jres.122.040