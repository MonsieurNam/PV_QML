NIST Technical Note 1896
High-Speed Monitoring of Multiple
Grid-Connected Photovoltaic
Array Configurations
Matthew T. Boyd
This publication is available free of charge from:
http://dx.doi.org/10.6028/NIST.TN.1896

NIST Technical Note 1896
High-Speed Monitoring of Multiple
Grid-Connected Photovoltaic
Array Configurations
Matthew T. Boyd
Energy and Environment Division
Engineering Laboratory
This publication is available free of charge from:
http://dx.doi.org/10.6028/NIST.TN.1896
October 2015
U.S. Department of Commerce
Penny Pritzker, Secretary
National Institute of Standards and Technology
Willie May, Under Secretary of Commerce for Standards and Technology and Director

Certain commercial entities, equipment, or materials may be identified in this
document in order to describe an experimental procedure or concept adequately.
Such identification is not intended to imply recommendation or endorsement by the
National Institute of Standards and Technology, nor is it intended to imply that the
entities, materials, or equipment are necessarily the best available for the purpose.
National Institute of Standards and Technology Technical Note 1896
Natl. Inst. Stand. Technol. Tech. Note 1896, 60 pages (October 2015)
CODEN: NTNOEF
This publication is available free of charge from:
http://dx.doi.org/10.6028/NIST.TN.1896

Preface
This effort was conducted by the Energy and Environment Division in the Engineering Laboratory at the National
Institute of Standards and Technology (NIST). This document describes the data acquisition systems (DASs) for
three photovoltaic (PV) arrays on the NIST campus in Gaithersburg, Maryland, USA, and includes the rationale
behind choices made for the sensors, data loggers, control software, and supplementary devices that comprise the
DASs. The intended audiences are modelers who wish to use the data gathered from these specific systems and
researchers who build and utilize similar DASs. Modelers and analysts who are interested in using the data in
collaboration on new research before a public data portal is made available can contact the author.
Author Information
Matthew T. Boyd
Mechanical Engineer
National Institute of Standards and Technology
Engineering Laboratory
100 Bureau Drive, Mailstop 8632
Gaithersburg, MD 20899-8632
Tel.: 301-975-6444
Email: matthew.boyd@nist.gov
ii

Abstract
Three grid-connected monocrystalline silicon photovoltaic arrays have been instrumented with research-grade
sensors on the Maryland campus of the National Institute of Standards and Technology (NIST). These arrays range
from 73 kW to 271 kW and are of different tilts, orientations, and configurations. Irradiance, temperature, wind, and
electrical measurements at the arrays are recorded, and images are taken of the arrays to monitor any shading or
anomalies. Nearly all measurements are sampled and saved every 1 second, with monitoring having started
August 1, 2014. This report describes the instrumentation approach to capture the performance of these photovoltaic
systems for use in computer model validation or performance monitoring.
Keywords
data acquisition, solar energy, photovoltaic array, inverter, meteorology
iii

Table of Contents
Preface .......................................................................................................................................................................... ii
Abstract........................................................................................................................................................................ iii
Table of Contents..........................................................................................................................................................iv
List of Figures ...............................................................................................................................................................vi
List of Tables ............................................................................................................................................................. viii
Glossary ........................................................................................................................................................................ix
1. Introduction ............................................................................................................................................................... 1
2. Overview ................................................................................................................................................................... 4
2.1. Summary ............................................................................................................................................................. 4
2.2. Canopy Array ..................................................................................................................................................... 5
2.3. Ground Array ...................................................................................................................................................... 8
2.4. Roof Array .......................................................................................................................................................... 9
3. Measurements .......................................................................................................................................................... 11
3.1. Summary ........................................................................................................................................................... 11
3.2. Irradiance .......................................................................................................................................................... 12
3.3. Ambient Temperature ....................................................................................................................................... 15
3.4. Module Temperature ........................................................................................................................................ 17
3.5. Wind ................................................................................................................................................................. 20
3.6. DC Electrical .................................................................................................................................................... 22
3.7. AC Electrical .................................................................................................................................................... 25
4. Data Acquisition and Control .................................................................................................................................. 27
4.1. Data loggers ...................................................................................................................................................... 27
4.2. Reference Modules ........................................................................................................................................... 36
4.3. Cameras ............................................................................................................................................................ 38
5. Data Archiving and Monitoring ............................................................................................................................... 39
6. Uncertainty .............................................................................................................................................................. 41
7. Maintenance ............................................................................................................................................................ 41
Acknowledgements ..................................................................................................................................................... 41
References ................................................................................................................................................................... 41
Appendix A. Oblique Images of the Arrays ............................................................................................................ A-1
Appendix B. Manufacturer Data Sheets ................................................................................................................... B-1
Appendix C. Copper Wire Gauges and Lengths ....................................................................................................... C-1
Appendix D. Diagrams of the Arrays ...................................................................................................................... D-1
iv

Appendix E. The Saved Data Values From Each Array ........................................................................................... E-1
Appendix F. Anonymized Diagram of the Data Acquisition Systems’ Communications Network ......................... F-1
v

List of Figures
Figure 1-1 A map of the instrumented PV arrays and weather station on the NIST campus in Gaithersburg,
Maryland. The horizontal roof array shown will be instrumented as part of a future project. The vertical
edge of this image corresponds to 1 km (0.6 miles or 3300 ft) .................................................................. 1
Figure 1-2 Canopy Array .............................................................................................................................................. 2
Figure 1-3 Ground Array .............................................................................................................................................. 2
Figure 1-4 Roof Array .................................................................................................................................................. 3
Figure 2-1 Diagram of the Canopy Array ice guards.................................................................................................... 6
Figure 2-2 Canopy Array electrical layout and measurement locations. The numbered squares are the combiner
boxes and the hatched regions are the source circuits. Note that “POA Pyra.” stands for a pyranometer
in the plane of the array, “GHI” stands for a pyranometer measuring the global horizontal irradiance,
and “RTD” stands for resistance temperature detector, which are mounted on the backsides of the
respective modules. The integrator also has an RTD mounted on the back of a cell on the module north
of combiner box 3, the same module with a NIST RTD. ........................................................................... 7
Figure 2-3 The Canopy Array inverter, identical to the Ground Array inverter ........................................................... 7
Figure 2-4 Ground Array electrical layout and measurement locations. The numbered squares are the combiner
boxes and the hatched regions are the source circuits. The gray regions on the front, back, and east sides
of the array are bioretention areas. Note that “POA Pyra.” stands for a pyranometer in the plane of the
array, “GHI” stands for a pyranometer measuring the global horizontal irradiance, and “RTD” stands for
resistance temperature detector, which are mounted on the backsides of the respective modules. The
integrator also has an RTD mounted on the back of a cell on the module east of combiner box 3, the
same module with a NIST RTD. ................................................................................................................ 8
Figure 2-5 Wind deflectors on the rear of the Roof Array module racks ..................................................................... 9
Figure 2-6 Roof Array electrical layout and measurement locations. The numbered squares are the combiner boxes
and the hatched regions are the source circuits. Note that “POA Pyra.” stands for a pyranometer in the
plane of the array, “GHI” stands for a pyranometer measuring the global horizontal irradiance, and
“RTD” stands for resistance temperature detector, which are mounted on the backsides of the respective
modules. ................................................................................................................................................... 10
Figure 2-7 The Roof Array inverter ............................................................................................................................ 10
Figure 3-1 The Canopy Array weather instruments at the north end of the center canopy, showing the irradiance
sensors and mounts as well as the ambient temperature sensor, wind sensor, and wind sensor surge
protector ................................................................................................................................................... 13
Figure 3-2 The Ground Array irradiance sensors at the center of the array, showing the sensors and mounts .......... 13
Figure 3-3 The Roof Array irradiance sensors at the southeast corner of the array, showing the sensors and mounts
................................................................................................................................................................. 14
Figure 3-4 The irradiance sensors at the center of the older horizontal PV array, showing the sensors and mounts . 14
Figure 3-5 The temperature probe and radiation shield used to measure the outdoor ambient air temperature at each
array ......................................................................................................................................................... 16
Figure 3-6 The temperature probe and enclosure used to measure the ambient air temperature in the inverters at the
Canopy and Ground Arrays ..................................................................................................................... 17
Figure 3-7 A visual comparison of the three types of module temperature sensors used at the arrays and for the
reference modules: the RTD used at the arrays (left), the RTD used for the reference modules (center),
and the thermocouple used for the reference modules (right). ................................................................. 18
Figure 3-8 An RTD mounted on the back of a module using thermally conductive epoxy and an adhesive film
overlay ..................................................................................................................................................... 19
Figure 3-9 The integrator’s module temperature sensor mounted on the back of a module at the Ground Array ...... 20
Figure 3-10 The wind sensor at the Ground Array ..................................................................................................... 21
Figure 3-11 The wind sensor and outdoor ambient temperature sensor at the Roof Array ........................................ 21
vi

Figure 3-12 The custom voltage divider circuit for measuring the array DC voltage ................................................ 22
Figure 3-13 The inverter DC combiner compartment at the Ground Array with the installed DC measurement
components, including the current shunts, and left-to-right on the upper DIN rail: the ambient
temperature sensor, shunt RTU, analog transmitter, voltage divider, and fuse holder ........................... 24
Figure 3-14 The outdoor DC measurement enclosure at the Roof Array, which includes the current shunts, and left-
to-right on the upper DIN rail: the fuse holder, voltage divider, analog transmitter, and shunt RTU .... 24
Figure 3-15 The AC power meter at the Ground Array (formerly branded by Square D), also showing the separate
dedicated enclosure for the voltage tap fuses ......................................................................................... 26
Figure 4-1 The data acquisition enclosure at the Canopy Array showing the data logger, measurement and
communication peripherals, battery backup, and power converters and distribution blocks ................ 27
Figure 4-2 The integrator’s RTU and a terminal block in their outdoor enclosure at the Ground Array ................... 29
Figure 4-3 Breakdown of the DAS energy draws and UPS backups .......................................................................... 30
Figure 4-4 The indoor communication hubs at the Ground Array for the NIST (top) and integrator (bottom)
equipment ............................................................................................................................................... 31
Figure 4-5 Diagram of local network at each array .................................................................................................... 32
Figure 4-6 Half-bridge for measuring RTDs .............................................................................................................. 35
Figure 4-7 The reference modules installed at the weather station ............................................................................. 36
Figure 4-8 The I-V curve tracer with maximum power tracker and load ................................................................... 37
Figure 4-9 The PTZ camera that records images of the Canopy Array ...................................................................... 38
Figure 4-10 The enclosure for the Canopy and Roof Array camera midspans and network switch ........................... 39
Figure 5-1 The main data dashboard for the Ground Array ....................................................................................... 40
Figure A-1 Oblique Aerial View of the Canopy Array Site ..................................................................................... A-1
Figure A-2 Oblique Aerial View of the Ground Array Site...................................................................................... A-2
Figure A-3 Oblique Aerial View of the Roof Array site .......................................................................................... A-3
Figure B-1 Manufacturer Datasheet for the PV Module at All Three Arrays ........................................................... B-1
Figure B-2 Manufacturer Datasheet for the Inverter at the Canopy and Ground Arrays .......................................... B-2
Figure B-3 Manufacturer Datasheet for the Inverter at the Roof Array.................................................................... B-3
Figure D-1 Diagram of the Canopy Array Strings and Pertinent Array Dimensions ............................................... D-1
Figure D-2 Diagram of the Ground Array Strings and Pertinent Array and Fence Dimensions .............................. D-2
Figure D-3 Diagram of the Roof Array Strings and Pertinent Array Dimensions .................................................... D-3
Figure F-1 Anonymized Diagram of the Data Acquisition Systems’ Communications Network ............................ F-1
vii

List of Tables
Table 2-1 Summary of the Campus PV Arrays ............................................................................................................ 5
Table 3-1 Summary of the Measurements and Respective Instruments Installed at the Campus PV Arrays ............. 11
Table 3-2 Inverter DC Measurements ........................................................................................................................ 23
Table 3-3 Inverter and Power Meter AC Measurements ............................................................................................ 26
Table C-1 Copper Wire Gauges and Lengths ........................................................................................................... C-1
Table E-1 The Saved Data Values From Each Array ............................................................................................... E-1
viii

Glossary
AC alternating current
ADC analog-to-digital converter
AWG American wire gauge
BORCAL Broadband Outdoor Radiometer Calibration
bps bits per second
comm. communication
CT current transformer
CW clockwise
DAS data acquisition system
DC direct current
DHI diffuse horizontal irradiance
DNI direct normal irradiance
DST daylight saving time
EMI electromagnetic interference
GHI global horizontal irradiance
IC integrated circuit
IR infrared
I-V current-voltage
LIU light interface unit
LST local standard time
NIST National Institute of Standards and Technology
NREL National Renewable Energy Laboratory
NZERTF Net-Zero Energy Residential Test Facility
POA plane-of-array
PoE Power over Ethernet
ppm parts-per-million
PTZ pan-tilt-zoom
PV photovoltaic
pyra. pyranometer
RTC Regional Test Center
RTD resistance temperature detector
RTU remote terminal unit
temp. temperature
UPS uninterrupted power supply
UTC Coordinated Universal Time
UV ultraviolet
VAC volts, alternating current
VDC volts, direct current
VDV Vista Data Vision
α temperature coefficient of resistance
Ω ohms
ix

1. Introduction
Three grid-connected solar photovoltaic (PV) arrays were constructed on the National Institute of Standards and
Technology (NIST) campus in Gaithersburg Maryland, with construction finishing around July of 2012. A map of
the sites is shown in Figure 1-1. Oblique aerial views of the three array sites are provided in Appendix A.
Figure 1-1 A map of the instrumented PV arrays and weather station on the NIST campus in Gaithersburg,
Maryland. The horizontal roof array shown will be instrumented as part of a future project. The
vertical edge of this image corresponds to 1 km (0.6 miles or 3300 ft)
The three arrays use the same monocrystalline silicon module, but all are mounted in different configurations: the
Canopy Array (Figure 1-2) is mounted on east/west facing canopies over a parking lot, the Ground Array (Figure
1-3) is mounted on tilted ground supports in an open field, and the Roof Array (Figure 1-4) is mounted on tilted,
weighted racks on a flat building roof.
1

Figure 1-2 Canopy Array
Figure 1-3 Ground Array
2

Figure 1-4 Roof Array
Research-grade sensors have been installed at each array and at the onsite weather station to measure various
electrical, temperature, and meteorological values. These data acquisition systems (DASs) were created to address
the needs for long-term, high accuracy, sub-minute data sets from large scale, calibrated, well-maintained and
documented systems in a variable environment, as expressed in [1], [2], [3], and [4]. The only other known arrays
instrumented to a comparable level with public to semi-public data sets in the northeast of the United States are the
65 kW to 907 kW arrays at the Northeast Solar Energy Research Center (NSERC) [5] and the 32 MW Long Island
Solar Farm [6], both installed at Brookhaven National Laboratory in Upton, New York, on Long Island, and the
6 kW and 60 kW Regional Test Center (RTC) arrays installed at an IBM facility in Williston, Vermont [7].
The weather station, at the location shown in Figure 1-1, measures various other meteorological quantities,
providing supplementary data to the PV data sets. These measurements include:
 direct normal irradiance (DNI)
 diffuse horizontal irradiance (DHI)
 global horizontal irradiance (GHI)
 infrared (IR) irradiance (net and total)
 ultraviolet (UV) irradiance (A, B, and total)
 spectral irradiance curves
 snow depth
 wind speed and direction
 humidity
 precipitation
 barometric pressure
 hail count
 ambient temperature
 whole sky images
The weather station DAS will be described in a forthcoming technical report.
3

The PV integrator who constructed the arrays has also installed a separate DAS at each PV array, monitoring a much
smaller scope of parameters. Hereafter, if a differentiation is not made between the systems, then the NIST system is
the one being referred to. At each array, both systems share communications with the inverter and AC power meter,
with the NIST system also recording all other measurements made by the integrator’s system.
The NIST systems save measurement values, including those from the integrator’s systems, every 1 second along
with the status of select components. Cameras near each array capture still images every 5 minutes, and a current-
voltage (I-V) tracer with a maximum power tracker and load saves curve traces from reference modules installed at
the weather station in the orientations of each array every 1 minute, along with one-minute averaged maximum
power measurements from one-second samples.
2. Overview
2.1. Summary
The arrays are located in Gaithersburg, Maryland in the United States, which is in the Eastern Time Zone (-5
hours from Coordinated Universal Time (UTC)). A summary of the arrays is given in Table 2-1. Location
information is for the center of the array, elevation is for the base of the array support (e.g., ground, roof) above sea
level, and height is the distance from the base of the array support to the bottom edge of the bottom module.
Manufacturer data sheets for the module and inverters are provided in Appendix B and the average lengths and
gauges of the wires are provided in Appendix C.
4

Table 2-1  Summary of the Campus PV Arrays
|                            | Canopy    |     | Ground    | Roof      |
| -------------------------- | --------- | --- | --------- | --------- |
| Array Rated DC Power [kW]  | 243       |     | 271       | 73        |
| Latitude [°N]              | 39.1385   |     | 39.1319   | 39.1354   |
| Longitude [°E]             | -77.2155  |     | -77.2141  | -77.2156  |
| Elevation* [m]             | 137       |     | 138       | 149       |
| Height** [m]               | 5.11      |     | 0.67      | 0.08      |
| Tilt [°]                   | 5         |     | 20        | 10        |
| Azimuth [° CW from N.]     | 90, 270   |     | 180       | 180       |
| Number of Modules          | 1032      |     | 1152      | 312       |
Sharp1
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
*  Elevation of the array support (e.g., ground, roof) above sea level
** Height of bottom edge of the bottom module above the array support (e.g., ground, roof)
2.2. Canopy Array
The Canopy Array in the parking lot northeast of NIST Building 227 is composed of five canopies running north-
south. The middle canopies have modules tilted nominally 5° east and west and the two end canopies have modules
tilted 5° toward the center of the array. Periodic local shading on the array is caused by the five-story Building 227
southwest of the array, a line of trees east of the array, and several trees south and southwest of the array. Some local
shading is also caused by ice guards along the outer longitudinal edges of the canopies that were designed to stop
large sheets of ice from falling off the array. A diagram of the ice guards is shown in Figure 2-1.

1 Certain commercial equipment, instruments, or materials are identified in this paper to foster understanding. Such
identification does not imply recommendation or endorsement by the National Institute of Standards and Technology, nor
does it imply that the materials or equipment identified are necessarily the best available for the purpose.

5

Figure 2-1 Diagram of the Canopy Array ice guards
The module strings in the array run predominantly north to south longitudinally in the five canopies. Individual
strings contain modules that are either all east facing or all west facing. These strings combine in parallel at seven
combiner boxes, at module height, to form seven output circuits that run to the inverter. Two of these output circuits
contain strings in the same canopies and orientations, three contain strings in the same canopies but in both
orientations, and two contain strings from four different canopies and in both orientations. This layout was designed
to keep modules that are in the same strings at the same tilts to avoid limiting their power output due to current
mismatch. This layout does, however, result in relatively longer source circuit wire lengths to the respective
combiner box DC disconnects, as the wires have to run down the 5.35 m (17.5 ft) high canopies and through 0.9 m
(3 ft) deep trenches and back up the other canopies. A diagram of the Canopy Array wiring layouts and the locations
of the combiner boxes, inverter, and all instruments and sensors is shown in Figure 2-2. An image of the inverter is
shown in Figure 2-3. A diagram of the string layouts and pertinent dimensions of the array is provided in Appendix
D.
6

Figure 2-2 Canopy Array electrical layout and measurement locations. The numbered squares are the
combiner boxes and the hatched regions are the source circuits. Note that “POA Pyra.” stands for
a pyranometer in the plane of the array, “GHI” stands for a pyranometer measuring the global
horizontal irradiance, and “RTD” stands for resistance temperature detector, which are mounted
on the backsides of the respective modules. The integrator also has an RTD mounted on the back
of a cell on the module north of combiner box 3, the same module with a NIST RTD.
Figure 2-3 The Canopy Array inverter, identical to the Ground Array inverter
7

2.3. Ground Array
The Ground Array in the field east of Building 216 is composed of five tilted rows, or sheds, running east-west. All
sheds are tilted at nominally 20° due south. The four front southernmost sheds are five modules wide while the back
north shed is four modules wide. Periodic local shading on the array is caused by inter-row shading, the three-story
Building 216 directly west of the array, vegetation in the bioretention area south of the array before it is periodically
trimmed, and a 1.2 m (48 in.) high cable rail fence surrounding the array. This fence, shown in Figure 1-3, has a 6.4
cm (2.5 in.) diameter horizontal top rail and 4.8 cm (1.9 in.) diameter vertical posts spaced 2.7 m (9 ft) apart.
The module strings in the array run east to west longitudinally in the five sheds. These strings combine in parallel at
seven combiner boxes approximately 1.5 m (5 ft) above the ground to form seven output circuits that run to the
inverter. Four of these output circuits contain strings in the same shed and the remaining three contain strings from
two different sheds. This layout does result in relatively longer source circuit wire lengths to the respective combiner
box DC disconnects, like the Canopy Array, as the wires have to run down the 2.06 m to 2.40 m (6.8 ft to 7.9 ft)
high ground supports and through 0.9 m (3 ft) deep trenches and back up the other sheds. A diagram of the Ground
Array wiring layouts and the locations of the combiner boxes, inverter, and all of the instrument and sensor locations
is shown in Figure 2-4. The inverter is identical to the one at the Canopy Arrays, as shown in Figure 2-3. A diagram
of the string layouts and pertinent dimensions of the array as well as dimensions of the fence are provided in
Appendix D.
Figure 2-4 Ground Array electrical layout and measurement locations. The numbered squares are the
combiner boxes and the hatched regions are the source circuits. The gray regions on the front,
back, and east sides of the array are bioretention areas. Note that “POA Pyra.” stands for a
pyranometer in the plane of the array, “GHI” stands for a pyranometer measuring the global
horizontal irradiance, and “RTD” stands for resistance temperature detector, which are mounted
on the backsides of the respective modules. The integrator also has an RTD mounted on the
back of a cell on the module east of combiner box 3, the same module with a NIST RTD.
8

2.4. Roof Array
The Roof Array is located on a one-story roof of Building 101 just south of the 14 story tower section of the
building. The array is composed of three sections arranged around various vents and other roof elements. The
modules are tilted nominally 10° due south; however, the north section of modules has a slight 0.5° clockwise
azimuthal offset. All of the module racks have an integrated sheet metal wind deflector at the back of the modules to
inhibit wind forces, as shown in Figure 2-5; however, these deflectors also inhibit convective cooling of the modules
from the wind.
Figure 2-5 Wind deflectors on the rear of the Roof Array module racks
Periodic local shading on the array is caused by inter-row shading, 1.4 m to 4 m (4.6 ft to 13 ft) high building
elements to the east and west of the array, the combiner boxes, and roof vents. The 14 story (13 stories above array)
building tower directly north of the array blocks a fraction of the diffuse irradiance but reflects and augments the
irradiance incident on the array during sunny conditions [8].
The module strings in the array run predominantly east to west. The two farthest east modules in the most southern
two rows are not connected to the array and will serve as replacements for any future broken modules in the three
arrays. The module strings combine in parallel at four combiner boxes 0.5 m (1.6 ft) above the roof to form four
output circuits that run to the inverter. Two of these output circuits contain strings in the same sections and the
remaining two contain strings from different sections. The sections in this array layout are relatively close together,
so there is significantly less additional wire needed to span the sections than that at the Canopy and Ground Arrays.
The inverter, however, is farther from the array than at the other two arrays, located approximately 60 m (200 ft)
north of the array in the tower section of the building at the same height of the array. A diagram of the Roof Array
wiring layouts and the locations of the combiner boxes and all of the instrument and sensor locations is shown in
Figure 2-6. An image of the inverter is shown in Figure 2-7. A diagram of the string layouts and pertinent
dimensions of the array is provided in Appendix D.
9

Figure 2-6 Roof Array electrical layout and measurement locations. The numbered squares are the combiner boxes and the
hatched regions are the source circuits. Note that “POA Pyra.” stands for a pyranometer in the plane of the array,
“GHI” stands for a pyranometer measuring the global horizontal irradiance, and “RTD” stands for resistance
temperature detector, which are mounted on the backsides of the respective modules.
Figure 2-7 The Roof Array inverter
10

3. Measurements
3.1. Summary
The measurements at each array and their corresponding sensor or instrument are summarized in Table 3-1. A
complete listing of the saved data values from each array is provided in Appendix E.
Table 3-1  Summary of the Measurements and Respective Instruments Installed at the Campus PV Arrays
| Measurement  | Instrument  | Make/Model  |
| ------------ | ----------- | ----------- |
Schneider Electric
AC Power Meter with Current
AC Voltage, Current, Freq., Power, Energy  Powerlogic ION8600 with
Transformers (CTs)
Unknown Make CTs
RTD Air Probe in Custom
| Ambient Temperature (inside inverter)  |     | Omega RTD-806  |
| -------------------------------------- | --- | -------------- |
Ventilated Enclosure
|     | RTD Probe in a Multi-Plate  | R. M. Young 41342LC  |
| --- | --------------------------- | -------------------- |
Ambient Temperature (outdoors) (1)  Passively Ventilated Radiation  in an
|     | Shield                      | R. M. Young 41003    |
| --- | --------------------------- | -------------------- |
|     | RTD Probe in a Multi-Plate  | Unknown Make Pt1000  |
Ambient Temperature (outdoors) (2)  Passively Ventilated Radiation  in an
|     | Shield  | Unknown Make Shield  |
| --- | ------- | -------------------- |
Campbell Scientific
| Current Signal  | Shunt Resistor  |     |
| --------------- | --------------- | --- |
CURS100
Voltage Divider in a Custom
| DC Voltage  |     | Caddock 1776-C6815  |
| ----------- | --- | ------------------- |
Circuit and Enclosure
| DC Current (1)  | Shunt Resistor  | Empro W-200-100  |
| --------------- | --------------- | ---------------- |
Empro HA/LAB-120-100
| DC Current (2)  | Shunt Resistor  |     |
| --------------- | --------------- | --- |
Empro HA/LAB-90-100
| Digital Image (1)  | PTZ Camera           | Axis Q6032-E  |
| ------------------ | -------------------- | ------------- |
| Digital Image (2)  | Campbell Scientific  | CC5MPX        |
Global Horizontal Irradiance (GHI)  Thermopile Pyranometer  Eppley PSP
Module Backsheet Surface Temperature (1)  RTD  Omega RTD-3-F3102-72-T
Module Backsheet Surface Temperature (2)  RTD  Omega SA1-RTD-4W-80
Module Backsheet Surface Temperature (3)  Thermocouple  Omega CO1-T-72
Module Backsheet Surface Temperature (4)  RTD  Unknown Make Pt1000
Plane-of-Array (POA) Irradiance (1)  Thermopile Pyranometer  Kipp & Zonen CMP 11
Flat-Plate Silicon Irradiance
| Plane-of-Array (POA) Irradiance (2)  |     | IMT Solar Si-420TC  |
| ------------------------------------ | --- | ------------------- |
Sensor
Domed Diffused Silicon-Cell
| Plane-of-Array (POA) Irradiance (3)  |     | Apogee SP-110  |
| ------------------------------------ | --- | -------------- |
Pyranometer
I-V Curve Tracer with
Reference Module I-V Curve and Maximum
|     | Maximum Power Tracker and  | Daystar Multi-Tracer 5  |
| --- | -------------------------- | ----------------------- |
Power Point
Load
Campbell Scientific
| RTD Current  | Shunt Resistor  |     |
| ------------ | --------------- | --- |
4WPB100
11

Shunt Voltage Remote Terminal Unit Gantner Q.bloxx A104
Campbell Scientific Inc.
Voltage Signal Data Logger
CR1000-ST
Advantech ADAM-4015
Voltage Signal and RTD Remote Terminal Unit
(custom)
Voltage Divider Output Voltage Analog Transmitter Acromag TT337
Wind Speed and Direction Ultrasonic Wind Sensor Vaisala WMT52
3.2. Irradiance
3.2.1. Sensors
The global horizontal irradiance (GHI) and the plane-of-array (POA) irradiance are measured at each array. The
GHIs are measured using Eppley Precision Spectral Pyranometers (PSPs) and the POA irradiances are measured
using both Kipp & Zonen CMP 11 pyranometers and IMT Solar Si-420TC flat-plate silicon irradiance sensors. Both
pyranometers are all-black thermopile, secondary-standard pyranometers, while the silicon sensor is a shunted
monocrystalline silicon cell under a thin glass cover with a cell temperature-compensated output signal.
Pyranometers were installed to most accurately measure the irradiance incident on the array in the two standard
orientations (horizontal and POA). Flat-plate silicon irradiance sensors were also installed to capture the irradiance
under fast changing conditions at nearly instantaneous speeds, unlike the pyranometers which have multiple-second
response times. The flat-plate irradiance sensors also better approximate the array’s effective irradiance, or that
absorbed by the PV cells (compared to the pyranometers which measure the incident irradiance,) because they have
similar reflective properties due to the flat glass covers and similar spectral responses since both utilize
monocrystalline silicon cells.
The PV integrator has also installed Apogee SP-110 domed diffused silicon-cell pyranometers at each array. These
instruments are measuring the POA irradiance, with the single pyranometer at the Canopy Array measuring the
irradiance in the east-tilted plane. These sensors function very similarly to the flat-plate irradiance sensors but have a
dome-shaped diffuser over the cell that minimizes reflective losses, which better approximates the incident, rather
than the absorbed, irradiance.
3.2.2. Calibrations
All of the thermopile pyranometers are calibrated at the National Renewable Energy Laboratory (NREL) according
to their Broadband Outdoor Radiometer Calibration (BORCAL) procedure [9]. This calibration provides net-IR
corrected responsivities (μV/W/m2) at every 2° in the measured solar zenith angle range. The POA pyranometers are
recalibrated every year while the GHI measuring pyranometers are calibrated approximately every (18 to 24)
months. The silicon sensors were calibrated in a solar simulator at the factory and verified to be within
specifications. The flat-plate irradiance sensors have a stated yearly drift of 0.5 %, and the domed diffused silicon
pyranometers have a yearly drift of less than 2 %.
3.2.3. Locations and Orientations
The locations of the irradiance sensors including those from the PV integrator are shown in the previous array
diagrams. The sensors are positioned in the actual module plane as near to the modules as possible without shading
them. As shown in the diagrams, none of the integrator’s pyranometers are co-located with other irradiance sensors
at the same tilt.
At the Canopy Array (Figure 2-2) there are two sets of POA irradiance sensors in both the east and west tilted
module planes. These sensors, along with the pyranometer measuring the GHI, are co-located at the north end of the
12

center canopy, as shown in Figure 3-1. This location is near the center of the array to measure the most
representative array irradiance but is on the north end of the array opposite the sun to avoid any shading on the
modules from the instruments. The integrator’s pyranometer is located at the south end of the center canopy in close
proximity to their outdoor data acquisition enclosure and positioned in the east-tilted module plane.
Figure 3-1 The Canopy Array weather instruments at the north end of the center canopy, showing the irradiance sensors and
mounts as well as the ambient temperature sensor, wind sensor, and wind sensor surge protector
At the Ground Array (Figure 2-4) there is a single set of irradiance sensors located in the middle of the array to
measure the most representative array irradiance, as shown in Figure 3-2. The integrator’s pyranometer is located at
the west end of the center shed in close proximity to their outdoor data acquisition enclosure and positioned in the
plane of the array.
Figure 3-2 The Ground Array irradiance sensors at the center of the array, showing the sensors and mounts
At the Roof Array (Figure 2-6) there is a set of GHI and POA irradiance measuring sensors in the southeast corner
of the array, as shown in Figure 3-3, located in an area that is least shaded by surrounding building and array
elements. Pyranometers measuring the GHI are also installed at the other three corners of the array to capture the
spatial variability of the irradiance caused by the 14 story (13 stories above array) building tower directly north of
13

the array. The building has a large number of windows and a light-colored brick façade with reflective aluminum
mullions (see Figure 1-4). The integrator’s pyranometer is located at the southwest corner of the array in relatively
close proximity to their outdoor data acquisition enclosure and positioned in the plane of the array.
Figure 3-3 The Roof Array irradiance sensors at the southeast corner of the array, showing the sensors and mounts
There are three more locations not pictured in the Roof Array diagram that are nearer to the building where the GHI
is also measured with pyranometers and flat-plate irradiance sensors. These additional sensors are situated around
and at the center of an older horizontal PV array, which is the subject of a study on aging that will be discussed in
later publications. The sensors at the center of this array are shown in Figure 3-4 and include two side-by-side flat-
plate irradiance sensors to quantify the level of soiling: the east one is regularly cleaned and the west one is not. The
irradiance measurements from these additional sensors are included in this dataset.
Figure 3-4 The irradiance sensors at the center of the older horizontal PV array, showing the sensors and mounts
3.2.4. Mounting and Alignment
The GHI and POA sensors are mounted using Campbell Scientific Inc. (CSI) CM245 Adjustable-Angle Mounting
Stands, as shown in previous figures, or mounted directly to an aluminum plate for the single location at the center
of the older horizontal roof array (Figure 3-4). The Eppley PSPs are mounted using nylon leveling and fastening
14

screws to avoid creating ground loops and the associated signal noise they can induce because the body of a PSP is
electrically connected to the ground pin on their cable connector. The GHI pyranometers are leveled using the
integrated bubble levels, ensuring that the bubble is in the middle of the center circle, with the bubble anywhere in
the circle corresponding to a leveling accuracy of approximately ±0.1° for the CMP 11’s and ±0.2° for the PSPs.
The GHI and POA flat-plate irradiance sensors are aligned to horizontal and the array nominal tilt, respectively,
using a Mitutoyo Pro 360 inclinometer to a reading within ±0.1°.
The alignments of the POA pyranometers are more challenging because there is no accessible flat surface on a
CMP 11 that is parallel to its sensor on which to rest an inclinometer. To align these instruments, a laboratory-grade
granite surface plate is first leveled to within a reading of ±0.1° using the inclinometer. Then, the pyranometers are
leveled on the surface plate using the pyranometers’ leveling feet and integrated bubble levels. The mounts for the
pyranometers are then aligned to a reading ±0.1° from the nominal array tilts using the inclinometer, and lastly the
pyranometers are fastened to them.
3.2.5. Wiring
The irradiance sensors are wired directly to the data logger multiplexers using UV and moisture protected cables run
through metal dedicated data conduit, separating them from power conductors that may generate high EMI. The
cables enter the conduit through cable glands at pull boxes or conduit access ports. The sensors at the Roof Array,
however, initially pass through shorter sections of nonmetallic conduit. The pyranometers are wired using shielded
twisted-pair cable grounded at one end on the data logger side to reduce and allow the electromagnetic interference
(EMI) to be later rejected when differentially measured, which is especially important for their ~10 µV resolution
analog voltage output. The flat-plate irradiance sensors are similarly wired using shielded twisted three-wire cable.
These silicon sensors communicate over an unpowered 4 mA to 20 mA current loop, which has a low sensitivity to
electrical noise, and runs on one pair, with the third wire supplying power to the temperature compensation and loop
transmitter circuits.
The integrator’s pyranometers are wired directly to their nearby outdoor data acquisition enclosures using the cable
pigtail connected to each sensor, entering through cable glands. These sensors communicate using an analog voltage
signal.
3.3. Ambient Temperature
3.3.1. Outdoor
The outdoor ambient air temperature is measured at each array using one R. M. Young 41342LC 1000 Ω platinum
RTD probe and transducer in an R. M. Young 41003 multi-plate passively-ventilated radiation shield, shown
disassembled in Figure 3-5. The radiation shields are used to shield the temperature probes from radiative heat
transfer from the sun and surroundings, which would affect the temperature of the probe, while still allowing
ambient airflow around the probe. Aspirated, or fan ventilated, radiation shields were not used due to backup power
considerations. The PV integrator has also installed a similar style two-wire, 1000 Ω RTD probe (without a
transducer) and radiation shield at each array of otherwise unknown specifications. Both sets of sensors were
calibrated at the factory and verified to be within specifications, and they have an unspecified yearly drift.
15

Figure 3-5 The temperature probe and radiation shield used to measure the outdoor ambient air temperature at each array
The sensors are located next to the arrays in locations that do not shade the modules, are not in stagnant air, and are
not near potentially heat-releasing sources like roof vents, with the specific locations of both sets of sensors shown
in the previous array diagrams. Only at the Ground Array are the NIST and integrator sensors co-located. Both sets
of sensors are mounted to poles and vertically aligned.
The ambient temperature sensors are wired directly to the data logger multiplexers using UV and moisture protected
cables, and at the Canopy and Ground Arrays these cables run through metal dedicated data conduit, separating them
from power conductors that may generate EMI. The cables enter the conduit through cable glands at pull boxes. The
sensor at the Roof Array is located immediately adjacent to the data acquisition enclosure, and its cable enters a
short nonmetallic conduit through a conduit access port. The sensors at all arrays are wired using shielded cable
grounded at one end on the data logger side to reduce EMI. These sensors communicate via a transducer over a
powered 4 mA to 20 mA current loop, which has a low sensitivity to electrical noise and also supplies power to the
sensor. The integrator’s ambient temperature sensing two-wire RTDs do not have transducers and are wired directly
to their nearby outdoor data acquisition enclosures, with the cables entering through cable glands.
3.3.2. Inverter
The ambient air temperatures inside the inverters at the Canopy Array and the Ground Array are each measured
using one Omega RTD-806 exposed, three-wire, class “A”, 100 Ω platinum RTD air probe with an α = 0.00385. The
exposed sensing element of these RTDs results in a fast response time. The RTDs are mounted inside a custom
perforated enclosure, as shown in Figure 3-6, with a fourth terminal split off from the three-wire RTD at the
enclosure to allow for more accurate four-wire measurements. The perforated enclosure mounts on a DIN rail and
physically protects the sensor and limits any radiative heat transfer from the surroundings, which would affect the
temperature of the probe, while still allowing ambient airflow around the probe. The sensors were calibrated at the
factory and verified to be within specifications, and they have an unspecified yearly drift. The ambient temperature
inside of the inverter at the Roof Array is measured by the inverter itself and is one of the measurements polled by
the data acquisition system, as discussed in a later section.
16

Figure 3-6 The temperature probe and enclosure used to measure the ambient air temperature in the inverters at the Canopy and
Ground Arrays
The sensors are located in the top center of the DC combiner compartment, with generally unseparated airflow to the
other inverter compartments including the AC and power module assembly compartments. The temperature sensors
are wired directly to the data logger multiplexers using UV and moisture protected cables run through metal
dedicated data conduit, separating them from power conductors that may generate EMI. The cables are also shielded
and grounded at one end on the data logger side to reduce EMI. The RTD cable has four wires and has the inner
wire insulation (and outer jacket) made of PTFE (Teflon). PTFE has a much lower permittivity and subsequent
capacitance than PVC and nearly all other wire insulation materials, resulting in faster settling times after excitation.
3.4. Module Temperature
3.4.1. Sensors
The module backsheet surface temperatures are measured at each array using Omega RTD-3-F3102-72-T exposed,
thin-film, four-wire, class “A”, 100 Ω platinum RTDs with an α = 0.00385. The sensing elements are flat, with only
a thin ceramic base material covering the resistive element, thereby reducing the thermal gradient in the sensor,
bringing it closer to the module surface temperature. The RTDs have relatively small sensing areas of 3.7 mm x
4.7 mm (0.147 in. x 0.185 in.) that reduce the insulating effect on the module, which affects its temperature local to
the RTD, while still providing enough area for adhesion and thermal contact. The relatively small size of the RTD
also results in faster response times due to the smaller thermal mass. A four-wire RTD was chosen instead of a two
or three-wire RTD because the long cable leads, the large changes in temperature and subsequent resistance of the
sensing element, and using a static, unbalanced resistance bridge would have resulted in significant measurement
errors.
The backsheet temperatures of the reference modules at the weather station are measured using a different RTD, the
Omega SA1-RTD-4W-80, which is very similar to the aforementioned RTD, but has a large 19 mm x 26 mm (0.75
in. x 1.0 in.) adhesive backing to the sensing element. This different RTD was chosen because the other model is
relatively delicate and rather difficult to install due to its small size. Thermocouples were also installed on the
reference modules, Omega model CO1-T-72. These thermocouples are ungrounded, flat, low mass, type T, ANSI
“Special Limits of Error” grade foil sensors with fast response times and high accuracy. These sensors were installed
to connect to the built-in thermocouple temperature channels on the I-V tracer with maximum power tracker and
load and provide a temperature measurement more easily correlated with the module measurements. A visual
comparison of the three temperature sensors is shown in Figure 3-7.
17

Figure 3-7 A visual comparison of the three types of module temperature sensors used at the arrays and for the reference
modules: the RTD used at the arrays (left), the RTD used for the reference modules (center), and the thermocouple
used for the reference modules (right).
RTDs were chosen instead of thermocouples or other temperature sensors for the primary module temperature
measurements because they are the most stable and accurate temperature sensing technology [10]. RTDs also do not
require a temperature reference as do thermocouples, which for the higher accuracy and many channel units can be
rather large, and in the case of ice point references require significant amounts of power. RTDs do, however, need
excitation, which requires a current source, increases the sampling time, and causes self-heating of the resistance
sensor that affects the measurement. These issues were addressed and minimized, as later described in the Data
Acquisition and Control section.
The PV integrator has also installed two-wire, 1000 Ω adhesive backed RTDs of otherwise unknown specifications
at each array that appear similar to the NIST adhesive backed RTDs.
3.4.2. Calibrations
All of the module temperature sensors were calibrated in the laboratory according to ASTM E220 [11]. According
to the standard, dry calibrations were performed, with the test and reference temperature sensors packed in brass test
tubes with thermally conductive aluminum oxide powder and immersed in a circulating temperature bath. With the
exception of one defective RTD, all RTDs and thermocouples were within specifications, and the proportional and
offset calibration coefficients and associated uncertainty were calculated for each sensor.
3.4.3. Locations
The locations of the temperature sensors including those from the PV integrator are shown in the previous array
diagrams. The sensors are mounted on the backsheets of the modules, with each sensor centered under a cell. Four
sensors are installed on one module in the center of each array and on each reference module at the weather station,
positioned according to IEC 60891 [12] (see previous array diagrams). The rest of the sensors in the arrays and the
additional thermocouples on the reference modules at the weather station are installed behind a center cell on
separate modules.
The sensor locations are distributed in the array with the intent of best capturing temperature gradients in the array
and best calculating a single representative array temperature. At the Canopy Array, nine temperature sensors plus
one from the integrator are installed, spanning the full length and half-width of the center canopy. It is assumed that
the other canopies will have similar temperature distributions. The integrator’s temperature sensor is co-located on
18

one of the same modules as a NIST temperature sensor, near the south end of the center canopy by the integrator’s
outdoor data acquisition enclosure.
At the Ground Array, nine temperature sensors plus one from the integrator are installed, spanning the full length
and width of the northwest quarter of the array. It is assumed that the four quarters of the array will have similar
temperature distributions. The integrator’s temperature sensor is co-located on one of the same modules as a NIST
temperature sensor, at the west end of the center shed by the integrator’s outdoor data acquisition enclosure.
At the Roof Array, seven temperature sensors plus one from the integrator are installed, spanning the full length and
width of the array. The integrator’s temperature sensor is not co-located with a NIST temperature sensor at this
array, and is located near the south end of the array by the integrator’s outdoor data acquisition enclosure.
3.4.4. Mounting and Alignment
The temperature sensors are mounted to the module backsheet using Omegabond OB-101 thermally conductive
epoxy and an Innotec 248DR, 57.2 mm (2.25 in.) diameter, 83 µm (3.3 mil) thick round polyester adhesive film
overlay, as shown in Figure 3-8. This method was derived as the combination of two of the best module back
surface temperature measurement methods as determined by Smith, et al. [13], specifically the methods designated
as positions C and H. This combination allows accurate measurement of the surface temperature, minimally affects
the surface temperature, results in a more precise bias from the true cell temperature, and provides reliably secure
attachment to the surface. The best practices recommended by Smith, et al. were also followed when mounting the
sensors, including minimizing the thickness of the epoxy under the sensor, minimizing the amount spread beyond
the sides of the sensor, and using additional adhesive films and ties on the cable for strain relief to prevent the sensor
from detaching. For the adhesive backed sensors, the thermally conductive epoxy is applied to the adhesive surface,
first removing the adhesive covers. The temperature sensors were aligned under the center of the respective cells by
distance measurements and visual means.
Figure 3-8 An RTD mounted on the back of a module using thermally conductive epoxy and an adhesive film overlay
The integrator’s sensors were mounted by the integrator, adhering the adhesive surface directly to the module
backsheet and overlaying with insulated foil tape, as shown in Figure 3-9. No thermally conductive epoxy was
employed.
19

Figure 3-9 The integrator’s module temperature sensor mounted on the back of a module at the Ground Array
3.4.5. Wiring
The module temperature sensors are wired directly to the data logger multiplexers using UV and moisture protected
cable run through metal dedicated data conduit, separating them from power conductors that may generate EMI. The
cables enter the conduit through cable glands at pull boxes or conduit access ports. The sensors at the Roof Array,
however, initially pass through shorter sections of nonmetallic conduit. The sensors are wired using shielded cable
grounded at one end on the data logger side to reduce EMI. The RTD cable has four wires and has the inner wire
insulation (and outer jacket) made of PTFE (Teflon). PTFE has a much lower permittivity and subsequent
capacitance than PVC and nearly all other wire insulation materials, resulting in faster settling times after excitation.
The RTDs are connected to the extension cables using waterproof IP67 connectors. The TCs are connected to the
extension cables using standard flat pin connectors housed in custom waterproof cable connector enclosures.
The integrator’s temperature sensors are wired directly to their nearby outdoor data acquisition enclosures using no
extension cables, entering through cable glands.
3.5. Wind
3.5.1. Sensor
The horizontal wind speed and direction is measured at each array using one Vaisala WMT52 heated ultrasonic
wind sensor. An ultrasonic sensor was chosen instead of a traditional vane and cup or windmill anemometer because
it is has a smaller physical profile, is easily heated and kept free of snow and ice, has no moving parts and therefore
a higher reliability, and has a higher sensitivity because of the faster response time and virtually zero starting
threshold (even very low wind speeds will be detected). The wind sensor is configured to sample readings at 4 Hz
and average every 1 s, and to perform maximum and minimum calculations instead of gust and lull. The wind sensor
controls its level of heating as a function of the ambient and its internal temperatures. The wind sensors
communicate over one wire pair using an RS485 serial ASCII protocol at 19200 bits per second (bps). Note that the
PV integrator did not install any wind sensors. The sensors were calibrated at the factory and verified to be within
specifications, and they have an unspecified yearly drift.
3.5.2. Locations and Orientation
The sensors are located next to the arrays in horizontal orientations in locations 56 cm to 150 cm (21.9 in. to 59.0
in.) above and 83 cm to 160 cm (32.5 in. to 62.8 in.) to the north of the modules. The wind sensors at the Canopy
(see Figure 3-1) and Ground Arrays (Figure 3-10) are north of the entire respective arrays while the sensor at the
Roof Array (Figure 3-11) is near the middle of the array, north of the middle section of modules. These locations
20

were chosen to measure the wind that is near to but out of the immediate boundary layer while minimally shading
the modules, which does occur but is limited to tens of minutes at sunrise and sunset in the summertime. The sensor
at the Roof Array is not north of the entire array like at the other arrays because it would shade the older horizontal
array north of this array and it would likely not be as representative of the wind over the array due to its closer
proximity to the 14 story (13 stories above array) building tower directly north of the array.
Figure 3-10 The wind sensor at the Ground Array
Figure 3-11 The wind sensor and outdoor ambient temperature sensor at the Roof Array
3.5.3. Mounting and Alignment
The wind sensors are mounted on top of vertical poles, aligned to true north using visual references and checked
using a compass (adjusted for the local magnetic declination). The wind sensors at the Canopy and Ground Arrays
are higher than any nearby sensor and structure so a Vaisala WSP150 surge protector, designed for this type of
application, was installed near each wind sensor in-line with the power and data cable to suppress power surges
caused by nearby lightning. Metal spikes designed for these specific sensors are installed on top of the sensor at each
array to deter birds from perching on them and affecting the readings.
21

3.5.4. Wiring
The wind sensors are wired to the surge protectors at the Canopy and Ground Arrays. These surge protectors and the
wind sensor at the Roof Array are wired directly to the data logger serial communication modules. All cables are
UV and moisture protected, and at the Canopy and Ground Arrays run through metal dedicated data conduit,
separating them from power conductors that may generate high EMI. The cables enter the conduit through cable
glands at pull boxes. The sensor at the Roof Array is located immediately adjacent to the data acquisition enclosure
and enters a short nonmetallic conduit through a conduit access port. The sensors are wired using shielded cable
grounded at one end on the data logger side to reduce EMI. These sensors communicate over one wire pair, with
power for the sensor and the heater on separate wire pairs. To reduce power consumption, termination resistors are
not wired in parallel at the ends of the RS485 communication wires to reduce signal reflections because they are not
needed, as the cable lengths are less than 600 m (2000 ft) and the data rates are relatively low (19200 bps). The
surge protectors for the wind sensors are grounded to local earth ground using a short length of 8 American wire
gauge (AWG) (8.37 mm2) wire. The wind sensor at the Roof Array is near higher building structures but is also
locally grounded like the other two wind sensors to prevent and mitigate surges caused by nearby lightning.
3.6. DC Electrical
3.6.1. Sensors
The direct current (DC) voltage of the arrays is measured at each array using one Caddock 1776-C6815 voltage
divider integrated circuit (IC) on a custom circuit board housed in a DIN rail mounted enclosure, as shown with the
cover off in Figure 3-12. Voltage dividers were chosen instead of other types of DC voltage transducers or sensors
because they are much more accurate and stable, with little long-term drift or sensitivity to their environment, and
therefore need no periodic adjustments to their gain and offset. Voltage dividers do however need to be connected to
very high input impedance (~10 MΩ) measurement devices to minimize current flow to the devices, which can
affect the measurements.
Figure 3-12 The custom voltage divider circuit for measuring the array DC voltage
The output circuit currents of the arrays are measured using seven Empro W-200-100 (0.5 mΩ) current shunts at
each of the Canopy and Ground Arrays and three Empro HA/LAB-120-100 (0.833 mΩ) and one HA/LAB-90-100
(1.111 mΩ) current shunt at the Roof Array. (These shunts are custom and have an HA style body and LAB style
base). These shunts were specified to best match the current range of the output circuits and the voltage range of the
isolating remote terminal units (RTUs) to which they are wired, and to be the most appropriate for the manner of
their installation. Current shunts were chosen instead of current transformers (CTs) or Hall effect sensors because
22

they are much more accurate and stable, with little long-term drift or sensitivity to their environment. Current
shunts, however, are often more difficult to install and may need more precautions than the other sensor options.
Isolators need to be used with both the voltage dividers and current shunts to electrically separate them from the
measurement circuits to prevent inadvertent secondary conductive paths from the array circuits to ground, which
would suppress the inverters’ ground fault detection. The voltage divider at each array is isolated with an Acromag
TT337, which is a high-voltage isolator and analog output transmitter. This transmitter was chosen because of its
high voltage isolation, high input impedance (15 MΩ), low uncertainties, and small form factor. The transmitter is
configured to sample readings at 13 Hz and employ various filters, including for rejecting 60 Hz noise. The
transmitter outputs a 0 V to 5 V signal proportional to the input voltage signal.
The current shunts at each array are isolated with a Gantner Q.bloxx A104 RTU, which is a high-voltage isolator
and 8-channel analog-to-digital converter (ADC). This RTU was chosen because of its high voltage isolation, small
form factor, and very low uncertainties at the millivolt levels output from the shunts, due partly to it being
specialized for measuring only millivolt signals, typically from thermocouples. The RTU samples readings at
100 Hz and averages every 10 ms. None of the configurable RTU filters are employed as tests showed that the RTUs
were more accurate and had a faster response time without them. The RTU communicates over one wire pair using
the RS485 serial Modbus RTU protocol at 19200 bps. A combination ADC and isolator RTU was chosen to
eliminate any EMI that would be picked up by the cable runs between the measurement location in the inverter or
shunt enclosure and the data logger. This EMI could have a relatively large effect on the millivolt shunt readings.
Use of the RTUs also resulted in less needed cables running back to the data loggers.
Various DC measurements are also performed by the inverters, as listed in Table 3-2. The inverters communicate
these measurements as well as the inverter operating state and any fault codes over one wire pair using the RS485
serial Modbus RTU protocol at 9600 bps. Note that the PV integrator did not install any DC measuring sensors, but
does communicate with the inverter.
Table 3-2 Inverter DC Measurements
Canopy and Roof
Measurement
Ground Inverters Inverter
PV voltage1 X X
Input voltage2 X X
Input current X X
Input power X X
Ground current X
Ground impedance X
1 The voltage on the array side of the DC contactor
2 The voltage on the load side of the DC contactor
3.6.2. Calibrations
The voltage dividers and isolators were calibrated in the laboratory and verified to be within specifications. The
current shunts were calibrated at the factory to an accuracy of ±0.10 %.
3.6.3. Locations and Orientations
The measurement components at the Canopy and Ground Arrays are located in the inverter DC combiner
compartment, as shown in Figure 3-13. The voltage divider and isolators are together at the top of the compartment
and the current shunts are horizontally oriented between the DC output circuit wires and the DC busbar.
23

Figure 3-13 The inverter DC combiner compartment at the Ground Array with the installed DC measurement components,
including the current shunts, and left-to-right on the upper DIN rail: the ambient temperature sensor, shunt RTU,
analog transmitter, voltage divider, and fuse holder
The sensors at the Roof Array are located exterior to the inverter in an outdoor enclosure, as shown in Figure 3-14,
approximately three-quarters of the wiring distance from the array to the inverter. The voltage divider and isolators
are together at the top of the enclosure and the current shunts are horizontally oriented in-line with the DC output
circuit wires. The measurements at the Roof Array were not made in the inverter because the manufacturer of that
inverter had recently ceased operations and therefore the installation design could not be reviewed.
Figure 3-14 The outdoor DC measurement enclosure at the Roof Array, which includes the current shunts, and left-to-right on
the upper DIN rail: the fuse holder, voltage divider, analog transmitter, and shunt RTU
3.6.4. Mounting
At the Canopy and Ground Arrays, the voltage divider and isolators are mounted to a DIN rail fastened to existing
threaded holes in the back panel in the DC combiner compartment. Also mounted on the DIN rail are the inverter
ambient temperature sensor in its enclosure (far left) and a fuse in a touch-safe holder (far right) for the DC voltage
tap. The current shunts are fastened to the front of the negative busbar using the same hardware, torque
24

specifications, and orientation for normally fastening the output circuit cables. The installation of the shunts does not
cause the separation between the negative and positive busbars to be reduced.
At the Roof Array, the voltage divider and isolators are mounted to a DIN rail fastened with screws to the enclosure
back panel. Also mounted on the DIN rail is a fuse in a touch-safe holder (far left) for the DC voltage tap (far right
under DIN rail). The shunts are also fastened with screws directly to the enclosure back panel.
3.6.5. Wiring
At the Canopy and Ground Arrays, high temperature, 600 V rated wires are connected to both the positive and
negative busbars using solder-filled ring terminals and the same hardware and torque specifications as that for the
output circuit cables, or in this case the current shunts. The positive wire is connected to the 0.5 A fast-blow fuse in
the touch-safe holder on the DIN rail, which is for disconnecting the voltage divider from the array if it fails and
becomes shorted. The other side of the fuse holder and the wire from the negative busbar are connected to the
voltage divider, which in turn is wired to its respective isolator. For the current measurements, the output circuit
cables are connected to the other side of the shunts in a similar manner as fastening the shunts to the busbar, with the
cables supporting all of their own weight due to their rigidity. High temperature, 600 V rated cables are connected to
the sensing terminals of the current shunts using solder-filled ring terminals and wired to their respective channels
on the isolating RTU.
At the Roof Array, a high-amp distribution block is connected in-line with one of the positive output circuit wires to
provide a positive voltage sensing terminal; a sensing terminal on one of the shunts provides the negative sensing
point. High temperature, 600 V rated wires are connected to both the positive and negative sensing terminals using
solder-filled ring terminals. The positive wire is connected to the 0.5 A fast-blow fuse in the touch-safe holder on the
DIN rail, which is also for disconnecting the voltage divider from the array if it fails and becomes shorted. The other
side of the fuse holder and the wire from the negative terminal (on a shunt) are connected to the voltage divider,
which in turn is wired to its respective isolator. For the current measurements, the negative current carrying wires
are cut and fastened to both sides of the respective shunts using compression lugs. High temperature, 600 V rated
cables are connected to the sensing terminals of the current shunts using solder-filled ring terminals and wired to
their respective channels on the isolating RTU.
At each array, the voltage divider transmitter and isolating RTU are wired directly to a data logger multiplexer and a
serial communication module, respectively. The inverter communication card is wired using Cat5 cable to a
communication hub in the closest building. To reduce power consumption, termination resistors are not wired in
parallel at the ends of the RS485 communication wires for the isolating RTUs and inverter cards because they are
not needed to reduce signal reflections, as the cable lengths are less than 600 m (2000 ft) and the data rates are
relatively low (19200 bps and 9600 bps, respectively). The cables are UV and moisture protected and run through
metal dedicated data conduit, separating them from power conductors that may generate EMI. The cables are also
shielded twisted-pair and grounded at one end on the data logger or indoor side to reduce and allow the EMI to be
later rejected when differentially measured.
3.7. AC Electrical
3.7.1. Sensors
The alternating current (AC) voltage, current, and other metrics are measured at each array using one Schneider
Electric Powerlogic ION8600 AC power meter. The meters use fused voltage taps (no potential transformers) and
revenue grade bar-type CTs with a 200:5 ratio at the Canopy and Ground Arrays and a 100:5 ratio at the Roof Array,
both of otherwise unknown specifications. Both the meter and the CTs were selected and installed by the PV
integrator. Various AC measurements are also performed by the inverters, with both the inverter and power meter
AC measurements listed in Table 3-3. The inverters communicate these measurements as well as the inverter
operating state and any fault codes over one wire pair using the RS485 serial Modbus RTU protocol, as described
25

before, and the AC power meters communicate over Ethernet using the Modbus TCP/IP protocol. Note that the PV
integrator also communicates with the inverter and AC power meter at each array.
Table 3-3  Inverter and Power Meter AC Measurements
|     | Canopy and  | Roof  | Power  |
| --- | ----------- | ----- | ------ |
Measurement
|                        | Ground Inverters  | Inverter  | Meter  |
| ---------------------- | ----------------- | --------- | ------ |
| Line voltage A,B,C     | X                 | X         | X      |
| Line current A,B,C     | X                 | X         | X      |
| Neutral current        |                   | X         |        |
| Output real power      | X                 | X         | X      |
| Output reactive power  |                   | X         | X      |
| Output apparent power  |                   | X         | X      |
| Power factor           |                   | X         | X      |
X1
| Total real energy  | X   | X   |     |
| ------------------ | --- | --- | --- |
X1
| Total reactive energy  |     |     |     |
| ---------------------- | --- | --- | --- |
X2
| Line frequency             | X   | X   |     |
| -------------------------- | --- | --- | --- |
| Phase reversal             |     |     | X   |
| Total harmonic distortion  |     |     | X3  |
1 includes separate received and delivered
2 includes minimum and maximum
3 maximum on each line
3.7.2. Locations
The AC power meters are installed in the closest building to each array near the building’s switchgear, with the
meter at the Ground Array shown in Figure 3-15. The CTs are housed in a junction box near the switchgear, and the
fused voltage taps are in the same junction box at the Roof Array and in one of the switchgear’s power distribution
boards at the Canopy and Ground Arrays, with the fuses housed in a separate dedicated enclosure near the AC
power meter.

Figure 3-15  The AC power meter at the Ground Array (formerly branded by Square D), also showing the separate dedicated
enclosure for the voltage tap fuses
26

3.7.3. Wiring
The inverters are connected to the 480 VAC, three-phase campus grid via a four-wire Y connection with the neutral
grounded. The power meters are connected to meter sockets wired to the buildings’ switchgear. The voltage taps are
connected through 2 A fuses and the CTs are wired to the meters using approximately 6 m to 9 m (20 ft to 30 ft)
leads. These leads are relatively long and their effect on the CT burdens and subsequent accuracy should be
considered. Each power meter is connected to the NIST network with a Cat5e cable, and each inverter
communication card, as explained before, is wired using Cat5 cable (twisted-pair) to a communication hub in the
closest building. The inverter communication cables are UV and moisture protected and run through metal dedicated
data conduit, separating them from power conductors that may generate EMI. The inverter cables are also shielded
and grounded at one end on the indoor side to reduce EMI.
4. Data Acquisition and Control
4.1. Data loggers
4.1.1. Components
The data logger at each array is a Campbell Scientific CR1000-ST, shown in Figure 4-1 in the upper left of the
Canopy Array outdoor data acquisition enclosure. The data acquisition enclosures at the Ground and Roof Arrays
are very similar and have identical components.
Figure 4-1 The data acquisition enclosure at the Canopy Array showing the data logger, measurement and communication
peripherals, battery backup, and power converters and distribution blocks
27

This data logger was chosen because it has:
 good measurement accuracy
 adequate processing power
 high reliability for extended periods in outdoor environmental conditions
 the ability to interface with a large variety of peripherals
 a small form factor
 relatively low power draws and therefore needs only a small battery backup for extended power outages
The analog voltage and current signals are multiplexed using one Campbell Scientific AM25T solid-state
multiplexer, with the current carrying wires first connected to Campbell Scientific CURS100 100 ohm current
shunts to convert the current signals to voltage signals. The RTDs are multiplexed using one Campbell Scientific
AM16/32B-ST relay multiplexer. The serial devices are each connected to separate Campbell Scientific SDM-SIO1
serial communication modules.
Solid-state multiplexers are used because of their faster switching times and greater longevity than relay
multiplexers, which are needed for measuring numerous channels at a one-second sample rate over an extended
duration. Relay multiplexers are used for the RTD measurements instead of solid-state multiplexers because the
solid-state multiplexers cannot pass the currents needed to excite the RTDs. This restriction does limit the RTD
sample rate to greater than one-second, but this rate is acceptable because the module temperatures change at a
slower rate than the irradiance and array DC voltage. The separate serial communication modules are used instead of
the serial communication channels on the data logger because there were not enough available channels.
The relay multiplexer is connected to the data logger through a Campbell Scientific 4WPB100 100 Ω four-wire half-
bridge terminal input module. The bridge employs a high-accuracy (±0.01 %), low temperature dependence (< 8
ppm/°C) shunt to measure the current through the RTD, generated by the excitation voltage supplied by the data
logger. A current limiting resistor is used in series with the shunt to allow for higher excitation voltages at the data
logger, minimizing the relative noise in the excitation, while limiting the current and subsequent self-heating in the
sensing resistor (RTD). Self-heating, which results in a measurement bias error, is caused by the excitation current
passing through the sensor resistor.
A Campbell Scientific NL115 is connected to the data logger’s peripheral port and provides 10BASE-T Ethernet
communications and a CompactFlash card slot interface for storing measurements. An industrial quality 2 GB
Synchrotech CFV-2GB-TSI-3524 card is used, which like the data logger, can operate with high reliability for
extended periods in outdoor environmental conditions.
The integrator’s RTU at each array is an Advantech ADAM-4015 RTD input module, shown in Figure 4-2 with a
terminal block in its outdoor data acquisition enclosure at the Ground Array. This RTU is specialized for measuring
RTDs, but the ones employed by the integrator are customized by the manufacturer to also measure analog voltages,
including the one from the integrator’s pyranometer. The RTU samples readings at 10 Hz, integrating each for
50 ms (three 60 Hz cycles), and does not average the readings. The RTU at each array is indirectly connected to the
integrator’s indoor data logger, which is an industrial computer of unknown specifications, communicating over one
wire pair using the RS485 serial Modbus RTU protocol at 9600 bps.
28

Figure 4-2 The integrator’s RTU and a terminal block in their outdoor enclosure at the Ground Array
4.1.2. Locations and Housing
Each data acquisition enclosure is located near the respective array in an easily accessible location that does not
shade the array. Each enclosure is NEMA 4 rated and locked to control access, with an additional sensor installed to
monitor when the door is open (at which time the data logger sends out an email alert). The humidity in the
enclosure is regulated by sealing the conduit openings with duct seal compound and replacing desiccant packets on
an as needed basis, with the humidity level monitored using a humidity indicator card. The enclosure at the Roof
Array is in a horizontal orientation to not shade the array, and is inside of another larger enclosure to better protect
against water ingress. The integrator’s outdoor RTU enclosures are also weather resistant and located near the array;
however, the enclosure at the Canopy array is located at the south end of the center canopy near the top of the
canopy and is only accessible with a ladder or lift.
4.1.3. Wiring and Power
All wiring inside of the enclosures is done with low capacitance, high temperature PTFE insulated stranded wires
with insulated ferrules crimped on their ends. Ferrule-terminated wires make troubleshooting and reworking easier
and provide a more secure, lower contact resistance connection to the terminals. Jumper wires are connected
between the low sides of the pyranometers’ differential inputs and the respective ground terminals to keep the
pyranometers, the only electrically floating sensors in these DASs, from floating above the ground potential and out
of the data logger’s common mode range, which would decrease the accuracy of the measurements. The jumpers
also keep the pyranometers from potentially floating out of the measurable range entirely.
The Ethernet communication module on the data logger connects via a Cat6 Ethernet patch cable to an Etherwan
EL1141 hardened 10/100BASE-TX Ethernet to 100BASE-FX fiber media converter. The media converter is
connected to a fiber light interface unit (LIU) using a fiber patch cable, where the LIU is a protective enclosure for
splitting and adding connectors to individual optical fibers. Fiber optic communication was chosen for the data
loggers to eliminate any EMI and reduce the potential for ground loops and power surges back to the data logger.
The distance between the data loggers and the network access point in the nearby buildings is also near to or over
the distance limit for Cat5/6 cable (100 m [330 ft]) which could have caused slow or unreliable communications.
The data logger is battery-backed using a Genesis NP24-12 12 V, 24 A∙h (288 W∙h) sealed rechargeable lead acid
battery connected to a Campbell Scientific CH200 charging regulator. This regulator charges the battery and
switches between line and battery power, providing an uninterrupted power supply (UPS) to the data logger and all
other peripherals in the enclosure except the Ethernet to fiber media converter. The UPS at each array will provide at
29

least 2.8 days of power to the DAS when there is no external power feeding the enclosure, even during high power
draw conditions (e.g., while heating the wind sensor in winter). A breakdown of the DAS energy draws is shown in
Figure 4-3.
Figure 4-3 Breakdown of the DAS energy draws and UPS backups
Power from the UPS is distributed to the data logger, peripherals, and an array of terminal blocks for the different
sensors via a Campbell Scientific A105 terminal expansion adapter on the charging regulator. The power from the
non-battery-backed power supply is distributed to a different array of terminal blocks, for currently just the Ethernet
to fiber media converter. The line inputs of the charging regulators for the UPSs at the Canopy and Roof arrays are
each connected to 120 VAC to 24 VDC power converters, while the terminal blocks for the non-battery-backed
Ethernet to fiber media converters are connected to 120 VAC to 12 VDC power converters. These two power
converters are connected to 120 VAC feeding those enclosures via terminal blocks and a single AC circuit breaker.
The charging regulator for the UPS at the Ground Array and the media converter are connected via terminal blocks
and a single AC circuit breaker to 24 VDC feeding that enclosure.
The data acquisition enclosure components at each array are grounded to a central point on the enclosure. Grounding
in a “star-point” configuration minimizes the potential differences between the devices and therefore reduces any
currents through the devices during a power surge event. The central point of the “star” is connected via the
enclosure to the conduit which is locally grounded. The conduit was chosen instead of the ground wires fed from the
nearby buildings because it provides a lower resistance path to ground and will not have an electrical choke effect
when there is a sudden, high current surge from a nearby lightning strike, which occurs for a surge carrying wire
inside of a ferrous conduit. The importance of proper grounding in the interest of equipment longevity cannot be
overemphasized.
Three conduits connect to each data acquisition enclosure, one carrying the AC or DC power wires or cable, one
carrying the fiber communication cable, and one carrying the sensor data cables. Excess lengths of the data cables
for potential future reworks (approximately 3 m [10 ft]) are stored in a nearby pull box connected to the data conduit
at the Canopy and Ground Arrays and in the outer enclosure at the Roof Array.
The integrator’s RTU is wired to the nearest building using Cat5 cable, communicating over one twisted-pair and
powered using two other twisted-pairs. At the Canopy Array, however, the RTU is powered by a NIST non-battery-
backed power supply in the outdoor NIST data acquisition enclosure. The data wires for this RTU are also
connected to a Moxa ISD-1230 RS485 surge protector next to the RTU in the integrator’s outdoor enclosure because
it has been very susceptible to failure, especially during storms. To reduce power consumption, termination resistors
30

are not wired in parallel at the ends of the RS485 communication wires because they are not needed to reduce signal
reflections, as the cable lengths are less than 600 m (2000 ft) and the data rates are relatively low (9600 bps).
4.1.4. Communications
The communication hubs for the data acquisition systems are at an indoor location in the nearest building to each
array. Two locked enclosures, one for the NIST equipment and one for the integrator’s data logger and RTU power
supply, are installed in the switchgear mechanical room near the AC power meter, as shown in Figure 4-4 at the
Ground Array.
Figure 4-4 The indoor communication hubs at the Ground Array for the NIST (top) and integrator (bottom) equipment
Near or inside the NIST enclosures are NIST network jacks with connections to the internal private network for the
NIST data loggers and to the public firewalled network for the integrator’s data loggers and all other equipment
accessed by both NIST and the integrator, including the AC power meters. When located outside of the NIST
enclosure, the network patch cables are locked to the jacks to prevent unauthorized disconnection. A complete
anonymized diagram of the communications network is shown in Appendix F, with a simplified diagram of the local
network at each array shown in Figure 4-5.
31

Figure 4-5 Diagram of local network at each array
The fiber optic cable from the outdoor data logger terminates in the NIST enclosure at a fiber LIU, which connects
to an Etherwan EL100 10/100BASE-TX Ethernet to 100BASE-FX fiber media converter. (At the Canopy Array,
these devices are located in a network closet on a different floor). The media converter connects to a network jack
on the internal private NIST network with a Cat6 patch cable. The Cat5 cables from the inverter and the integrator’s
RTU at each array terminate at separate Moxa ISD-1230 RS485 surge protectors in the NIST enclosure. These,
along with the other equipment in the enclosure, are grounded using 10 AWG (5.26 mm2) wire in another star-point
configuration to a central point in the enclosure that is connected to the building ground.
The outputs from the surge protectors are wired to a CommFront HUB-485-4 RS485 hub. The single input on the
hub is wired along with the serial communication port on the integrator’s data logger to separate serial ports on a
Moxa MGate MB3270 serial-to-Ethernet Modbus gateway, both through additional, smaller SerialComm ISO-485-P
surge protectors. This gateway translates and coordinates communications between the inverter and integrator’s
RTU and both the NIST data logger (Ethernet connection) and the integrator’s data logger (serial connection). The
gateway connects to a network jack on the public firewalled NIST network with a Cat6 patch cable. The NIST data
loggers communicate over the NIST network with the inverters and RTUs through the gateways, using the Modbus
TCP/IP protocol over Ethernet, which is translated by the gateways into the RS485 serial Modbus RTU protocol,
and vice versa.
The NIST indoor communication enclosure at the Ground Array also contains a 24 VDC power supply for the
outdoor data logger and a Power over Ethernet (PoE) midspan for the Ground Array camera. The NIST enclosure at
the Roof Array contains an extra smaller surge protector for connecting the RS485 hub to RTUs measuring module
temperatures sensors on the older horizontal PV array.
32

4.1.5. Configuration
4.1.5.1. Sampling Rates and Parallel Processing
The data loggers are programmed using the Campbell Scientific language CRBASIC. The programs measure all of
the sensors every 1 s, except for the RTDs which are measured every 10 s. The slower RTD sampling rate is due to
the slower relay multiplexers to which the RTDs are connected, but is not a restriction nonetheless because the
module temperatures the RTDs are measuring are not changing appreciably faster than every 10 s. It takes longer
than 1 s to measure, and poll and receive data from the sensors and instruments if done sequentially, so the program
is run in “pipeline” mode which effectively executes some of the measurement steps in parallel.
4.1.5.2. Time Synchronization
In order to maintain time synchronization within 1 s between data loggers, the data loggers are instructed every
morning just after midnight to synchronize with a local internet time server. This synchronization keeps the data
loggers within 0.5 s of each other at all times. The data loggers are configured to always use local standard time
(LST) to eliminate issues arising from the switch to and from daylight saving time (DST) and to allow easier
comparisons and calculations.
4.1.5.3. Channel Ranges
The input voltage ranges for each measurement channel are explicitly chosen and fixed; auto-range is not used to
eliminate the initial measurement needed to set the range, resulting in a shorter sampling time. The range determined
by auto-range may also not be accurate for the irradiance and electrical measurements as they can fluctuate rapidly.
4.1.5.4. Self-Calibration
The data logger is instructed to periodically self-calibrate to compensate for aging-induced and operating
temperature-induced (Seebeck effect) drift. Voltage offsets are measured on each channel when the signal source is
internally disconnected, using an internal voltage standard as a reference for all channels.
4.1.5.5. Measurement Techniques
Various techniques are employed for every measurement to more accurately measure the sensed value. The voltage
signals, both direct from the sensor and those from a shunt when measuring current signals, are measured
differentially instead of single-ended with respect to the data logger ground. This approach eliminates any voltage
reference offsets and rejects capacitively coupled common-mode noise.
Although considered, the thermopile pyranometers, the only electrically floating sensors in these DASs, are not
pulled to ground before each measurement to null any common-mode voltage. Rather, jumper wires are connected
between the low sides of the differential inputs for the pyranometers and the respective ground terminals to keep the
pyranometers grounded, as previously described. These data loggers combine nulling the common-mode voltage
with open input detection, commonly used for thermocouples, into one function and neither can be individually
implemented. The open input detection, which biases the high side of the differential input for the pyranometer
above the channel measurement range while grounding the low side, would not allow the pyranometer output
enough time to stabilize before the measurement. No measurement channels in the DASs employ the common-mode
nulling and open input detection function.
The settling times, the delay after the channel terminals are internally connected to the input amplifier and before the
measurement is started, were chosen to be 10 ms for the pyranometers and 3 ms for other analog signals. The longer
settling time for the pyranometers allows further reduction of voltage transients caused by channel switching, as they
would have a larger relative magnitude compared to the pyranometers’ much smaller voltage signals. Due to the
larger number of sensors at the Roof Array, only the essential pyranometers in the southeast corner of the array and
in the center of the older horizontal array are allowed to settle for 10 ms, with the rest settling for 3 ms, to allow for
all sensors to be sampled every 1 s.
33

The voltage signals are integrated for exactly 16.67 ms. This integration time, the period of a 60 Hz waveform,
rejects 60 Hz frequency components in the signal by cancelling them out. This noise is induced from nearby AC
power sources, which operate at 60 Hz in the USA. The signals for the nonessential pyranometers at the Roof Array
are integrated for 250 µs instead of 16.67 ms, also to allow for all sensors to be sampled every 1 s.
Each differential voltage measurement is performed twice, with the second time after the channel terminals are
internally reverse-connected to the input amplifier. The two measurements are subtracted and divided by two to
cancel any voltage offsets in the measurement circuitry, as well as any common-mode gain errors. Note that before
each measurement the signal is allowed time to settle, which doubles the total settling time per measurement.
There is not enough available time in the 1 s scan period to employ all of the aforementioned techniques when
measuring the four-wire RTDs, so tests were performed to identify those most beneficial in this application. The
tests determined that a settling time of 3 ms was adequate and revealed that 60 Hz integration reduces significantly
more uncertainty than reversing the inputs and reversing the excitation, combined. Reversing the excitation is
similar to reversing the inputs, but does so for the excitation and works to cancel any voltage offsets before and in
the measurement circuitry (reversing the input cancels the offset in the measurement circuitry as well as the
common-mode gain errors). Note that these two techniques each double the total settling time and combining both
quadruples it. If there was a longer sampling period then all three techniques would have been employed, even
though the benefits of the two reversals are relatively small in this application.
The excitation for the RTDs is optimized by maximizing the measurement resolution of the voltage across the
sensing resistor (RTD) while keeping the self-heating error, caused by the excitation current passing through the
sensing resistor, below an acceptable level. Self-heating errors for 100 Ω RTDs like the ones in these DASs are less
than 0.001 °C (0.0018 °F) in still air when the voltage drop across the sensing resistor is less than approximately 25
mV (6 µW) [14]. The optimal excitation voltage is therefore what results in a maximum sensor voltage of 25 mV,
which occurs at the highest expected temperature, chosen conservatively to be 90 °C (194 °F). This corresponds to a
maximum resistance of 135 Ω. The optimal excitation voltage is calculated for the half-bridge and current-limiting
resistor in Figure 4-6 using Equation (1), neglecting the negligible lead resistances. The equation is calculated using
values of the measurement bridge resistors at the low end of their respective tolerance ranges, resulting in a higher
current and subsequently higher RTD voltage. If the tolerances are not accounted for, this potentially higher RTD
voltage could be greater than the selected measurement range.
34

Figure 4-6 Half-bridge for measuring RTDs
V
V  s,max  R R R  (1)
x R cl,min sh,min s,max
s,max
where:
V = excitation voltage
x
V = sensing voltage
s
R = sensing resistor (RTD)
s
R = current limiting resistor
cl
R = completion (shunt) resistor
sh
V = shunt voltage
sh
I = excitation current
x
The optimal excitation, calculated to be 1.8 V, is only kept on during the measurement to further reduce the self-
heating errors. The measurement equation for the half-bridge, when reduced, becomes the ratiometric relation
R R V V . The temperature of the RTD is then determined from the resistance using the inverse of the
s sh s sh
Callendar-Van Dusen equation according to the ASTM E1137-04 [15] standard.
Timeouts for the serial and Ethernet communicating instruments, the time after polling the instrument waiting for a
reply, were carefully chosen to be the maximum time the instrument normally takes to respond to a request. A flag
in the data logger programs can be manually set while the program is running to stop polling individual instruments
in the case where they are temporarily unresponsive, which has occurred for the integrator’s RTUs and the inverters.
Setting this flag can speed up the program and avoid missed measurements while the data logger would normally be
waiting the full timeout period for a response.
4.1.5.6. Control
The data loggers regulate the power to the inverter shunt RTUs and the wind sensor heaters when external power to
the data acquisition enclosures is off and the data loggers are operating on battery power. These two devices
consume the most power and are regulated to allow the DASs to operate for approximately three days on battery
35

power. Power to the inverter shunt RTUs is turned off at night between 20:30 and 04:00 LST. Power to the wind
sensor heater is turned on during the first five minutes of each hour and turned off at all other times.
The data logger at the Roof Array, due to its high utilization, drops its communications every few days and needs a
power reset to restore it. One of the data loggers that is part of the weather station remotely toggles the power to the
Roof data logger via a relay every other day at 01:00 LST or when the data logger is unresponsive. The relay is
controlled over the public network through the RS485 hub at the Roof Array.
4.1.5.7. Data storage and processing
Measurements are stored on the memory card as either 16-bit floating point, 32-bit floating point, or 32-bit signed
integer data types according to the needed range and precision of the individual measurements. This selective
allocation conserves storage space and increases processing speeds in the data logger and server. All measurements
are saved every one-second, including the RTD measurements, resulting in nine duplicate RTD measurements saved
every 10-second sample period. All measurements are also either averaged or the minimum or maximum calculated
for each minute, whichever more appropriate, and saved. The one-minute values are calculated from the samples
taken between 1 s and 60 s, inclusive, preceding the recorded minute time. The wind measurements are processed
according to EPA guidelines [16] to produce the mean horizontal wind speed, the unit vector mean wind direction,
and the standard deviation of wind direction calculated using the Yamartino algorithm.
The 2 GB of storage can store 82 days to 131 days of measurements depending on the array for times when the
communications are down. The storage is configured as ring memory, where the oldest data is overwritten when the
memory card is full.
4.2. Reference Modules
Reference modules, one for each orientation at each array, are installed in the same orientations at the weather
station, as shown in Figure 4-7. These reference modules provide baseline measurements without the significant
wiring, mismatch, soiling, snow, shading, and other losses of the modules in the arrays. These reference modules
also provide data on their aging, and their true average cell temperature and absorbed/effective irradiance from their
measured I-V curves, which can be correlated with the measured backsheet surface temperature and POA irradiance,
respectively.
Figure 4-7 The reference modules installed at the weather station
There are two modules for the Canopy Array (facing left and right), one module each for the Ground Array (back
left) and Roof Array (second from front), one module for the older horizontal array near the Roof Array (front), and
36

one module for the array on NIST’s Net-Zero Energy Residential Test Facility (NZERTF) (back right). These
reference modules (except the one for the older horizontal array) are from the same production batches as those
installed at each array. The modules are arranged according to a CAD model made in SketchUp [17], which
calculated less than 45 minutes of any mutual or external shading on the modules at sunrise and sunset on the
summer solstice and less than 5 minutes on the winter solstice, the two extremes of the year. Flashing is installed
around the modules for the older horizontal array, Roof Array, and NZERTF array to simulate the effect of the
respective installations on the module temperatures; the modules in the Canopy and Ground Arrays have open air
circulation on their back sides and therefore have no flashing installed around their reference modules.
The backsheet surface temperatures of each module are measured using one Omega CO1-T-72 thermocouple
mounted behind a center cell and either four Omega SA1-RTD-4W-80 RTDs for the Canopy, Ground, Roof Array
modules or four additional Omega CO1-T-72 thermocouples for the older horizontal and NZERTF array modules,
matching the sensor type at each array. The set of four temperature sensors are mounted behind one center and three
peripheral cells, as described in section 3.4. The POA irradiance is measured for each module using an IMT Solar
Si-420TC flat-plate silicon irradiance sensor, as described in section 3.2.
The modules, thermocouples, and irradiance sensors are connected to a Daystar Multi-Tracer 5 (MT5) I-V curve
tracer with maximum power tracker and load system, as shown in Figure 4-8. (The set of four temperature sensors
on each module are connected to a data logger multiplexer for the weather station, sampled every 10 s and saved
every 1 s, and averaged and saved every 1 min.) The modules are wired directly to the system using 10 AWG (5.26
mm2) UV and moisture protected PV cable run through metal dedicated conduit. The sensors are wired as described
in sections 3.2.5 and 3.4.5.
Figure 4-8 The I-V curve tracer with maximum power tracker and load
37

The I-V tracer and maximum power tracking system performs an I-V trace for each module every minute,
integrating every current and voltage reading for 100 µs. The curves are saved along with measurements of the
respective thermocouple and irradiance sensor before and after each trace. At all other times, the system maximum
power tracks each module, measuring the maximum power voltage and current and the corresponding thermocouple
and irradiance sensor, sampling every one second and averaging and saving every minute. All measurements are
saved to internal memory. The MT5 system does not adjust its clock for DST, which allows for easier comparison
with the measured data from the data loggers, which also do not adjust for DST. The system was calibrated at the
factory and verified to be within specifications.
4.3. Cameras
Images of each array are taken using an Axis Q6032-E pan-tilt-zoom (PTZ) network camera located at each array,
with the camera for the Canopy Array shown in Figure 4-9. The images taken by the cameras are used to capture the
shading, snow, rain, and frost, and any other obstructions or damage to the arrays.
Figure 4-9 The PTZ camera that records images of the Canopy Array
The cameras are configured for constant white balance, auto exposure, and to use an IR cut filter during the day but
not at night to increase sensitivity. The cameras, like the data loggers, are configured to not adjust their clocks for
DST for easier comparison with the measured data. The cameras take 704 x 480 resolution images, which are saved
as uncompressed JPEG images on an internal 32 GB SanDisk Extreme SD card. Each camera is programmed to take
images of the entire array and of close-up sections at a constant zoom for detailed analysis and stitching together
into a single higher resolution image. The cameras take images every five minutes during the day (04:00 to 20:00
LST) and every hour all day.
38

The array cameras communicate over the NIST private network and are powered using Power over Ethernet (PoE).
A surge protector is housed in an outdoor enclosure, and the midspan, for combining the power and Ethernet, and
any necessary network switch and power supply are housed in an indoor enclosure, with the enclosure for the
Canopy and Roof Array cameras shown in Figure 4-10.
Figure 4-10 The enclosure for the Canopy and Roof Array camera midspans and network switch
There is also a Campbell Scientific CC5MPX camera at the weather station that takes images of the reference
modules. This camera does not adjust for DST and takes 2592 x 1944 resolution images saved as uncompressed
JPEG images on an internal 32 GB SanDisk Extreme SD card. No other settings are configurable. The camera is
programmed to take a fixed zoom image every five minutes during the day (04:00 to 20:00 LST) and every hour all
day. This camera communicates over the NIST private network and is powered using a separate cable.
5. Data Archiving and Monitoring
Campbell Scientific LoggerNet Admin is used to manage the network of data loggers. It is installed on a network
server running Windows Server 2008 R2. Loggernet Admin runs as a service, running whether or not a user is
logged in and starts immediately after the server restarts. Loggernet is configured to import data from the data
loggers every minute, leaving a copy on the data loggers as a backup to be later overwritten. The data are saved on
the server as Campbell Scientific TOA5 formatted plain text files, one for each of the 12 defined data logger tables
from the Canopy, Ground, and Roof Array data loggers, including the one-minute aggregated tables but not
including the file of the temperature measurements of the reference modules from the weather station data logger.
39

The data files are imported to a MySQL database every minute by Vista Data Vision (VDV) Enterprise by Vista
Engineering, also installed on the server. VDV is configured to send email alerts when data values are outside of
defined ranges and when the data files are not being updated. Custom dashboards for each array with plots of select
measurements are updated by VDV every minute and provide an overview of the system health. These dashboards
are displayed in a web interface hosted by the server, with the main data dashboard for the Ground Array shown in
Figure 5-1. In addition to the dashboards, custom plots of various types, tables, logs of alarms, completeness of the
database, and a viewer for the hourly camera images of each array are also provided in the web interface. This web
interface is currently only available inside the NIST firewall, but a public data portal with the ability to download
data will be made available. In the meantime, interested modelers and analysts who would like to use these data in
collaboration on new research can contact the author.
Figure 5-1 The main data dashboard for the Ground Array
Data from the I-V curve tracer and maximum power tracking system and images from the cameras are copied to the
server every 30 minutes by an FTP client on the server run as a scheduled task. A custom script, also run as a
scheduled task, monitors and sends alerts when data files from the I-V curve tracer and maximum power tracking
system are not being transferred to the server. VDV monitors and sends alerts when new images from each camera
are not being transferred. All data and images on the server, including data from the data loggers, are continually
backed up to offsite cloud storage.
40

6. Uncertainty
Uncertainties for the collected data are not listed because a complete uncertainty analysis is outside the scope of this
report, but one will be performed for each measurement and presented in a forthcoming paper. Sensor and
instrument-level uncertainties are not given as they are only part of the system-level values, and therefore can be
misleading. Furthermore, they do not account for the error between the measurand, the quantity of interest, and the
realized quantity, the quantity measurable by the sensor.
7. Maintenance
The data dashboards for each array are checked every weekday for any apparent problems in the DASs. The
irradiance sensors at the Ground and Roof Arrays are cleaned every other day, minus weekends, using cleanroom
wipes moistened with a 6 % isopropyl/94 % deionized water mixture and dried with optical tissue. Any ice buildup
is first melted using an ethanol solution of the same concentration. One of the flat-plate irradiance sensors in the pair
measuring the soiling at the Roof Array is never cleaned. The irradiance sensors at the Canopy Array are cleaned
only two to three times a year due to difficulty in gaining access to the elevated instruments, but a solution is being
implemented that will allow regular cleaning. A log is kept of the dates and times when the sensors at the different
arrays are cleaned.
The data files on the server grow indefinitely, so once a week a custom script is run that pauses the data importation
from the data loggers and runs the Campbell Scientific program File Format Convert that splits the files into
separate daily files for easier archiving and use.
Thermopile pyranometers are replaced every 12 to 18 months with newly calibrated pyranometers from NREL. A
log is kept of these instrument replacements, which also contains all other maintenance including when sensors,
instruments, and devices fail and are replaced and when the arrays are offline.
Acknowledgements
The author wishes to thank all those who contributed time and effort planning, constructing, and implementing the
infrastructure needed for the data acquisition systems described in this report. They include Bill Robson and John
Selby of Legatus 6, Christoph Heinzer of Advanced Energy, Carrie Kelly of NIST’s Office of Facilities and
Property Management, and Mike Zelinsky of SunEdison. Special thanks to Brian Dougherty of NIST’s Engineering
Laboratory’s (EL’s) Energy and Environment Division for logistical, construction, and maintenance support. Thanks
to Luis Luyo, also of EL’s Energy and Environment Division, for constructing the infrastructure for the reference
modules, and to Jeffrey Sze for early testing of the systems. Thank you also to Ryan Smith of Pordis, LLC, then at
NREL, for early instrument selection advice and design resources.
References
[1] K. Lynn, SunShot, (2011). http://www.solar.energy.gov/pdfs/doe_lynn_pv_validation_2011_aug.pdf
(Accessed on Sept. 11, 2015)
[2] J. Granata, PV performance and reliability validation capabilities at Sandia National Laboratories and the
National Renewable Energy Laboratory, (2011).
http://energy.gov/sites/prod/files/2014/01/f7/pv_manufacturing_workshop_2011_march_granata.pdf (Accessed
on Sep. 11, 2015)
[3] J. Newmiller, Validation information for PV power system project review, (2011).
http://www.solar.energy.gov/pdfs/dnv_newmiller_pv_validation_2011_aug.pdf (Accessed on Sep. 11, 2015)
[4] Sandia National Laboratories, PV validation and bankability workshop - August 31, 2011 - Survey Results,
(2011). http://energy.gov/sites/prod/files/2014/01/f7/pv_vb_surveyresults.pdf (Accessed Sep. 23, 2015)
[5] Brookhaven National Laboratory, Northeast solar energy research center (NSERC) - A multi-purpose research
facility on the BNL campus, (2013). https://www.bnl.gov/energy/files/nserc/NSERCbro3.pdf (Accessed on
Sep. 11, 2015)
41

[6] Brookhaven National Laboratory, Overview of BNL’s Solar Energy Research Plans, (2011).
http://www.bnl.gov/energy/files/nserc/BNL_Solar_Research_Overview_and_NSERC_Plans_with_Input.pdf
(Accessed on Sep. 9, 2014)
[7] Sandia National Laboratories, Vermont regional test center, (2014). https://rtc.sandia.gov/wp-
content/uploads/2014/08/7-VT-RTC_SAND2014-4204P.pdf (Accessed on Sep. 11, 2014)
[8] M. Boyd, Analytical model for solar irradiance near a planar vertical diffuse reflector – Formulation,
validation, and simulations, Sol. Energy 91 79–92 (2013).
[9] I. Reda, D. Myers, and T. Stoffel, Uncertainty estimate for the outdoor calibration of solar pyranometers: a
metrologist perspective, NCSLI Meas. 3 (4), 58–66 (2008).
[10] Omega Engineering, Temperature handbook and encyclopedia, (2010).
[11] ASTM, Standard Test Method for Calibration of Thermocouples By Comparison Techniques, ASTM
International, (2007).
[12] IEC, Photovoltaic devices – Procedures for temperature and irradiance corrections to measured I-V
characteristics, International Electrotechnical Commission (IEC), Geneva, Switzerland(2009).
[13] R.M. Smith, S. Kurtz, and B. Sekulic, Back-of-module temperature measurement methods, SolarPro 90–104
(2011).
[14] Campbell Scientific, CR1000 Measurement and Control System, (2013).
https://s.campbellsci.com/documents/us/manuals/cr1000.pdf (Accessed Sep. 23, 2015)
[15] ASTM, Standard Specification for Industrial Platinum Resistance Thermometers, ASTM International, (2008).
[16] EPA, Meteorological Measurements, Quality Assurance Handbook for Air Pollution Measurement Systems,
(2008).
[17] SketchUp Pro, Trimble Navigation Limited, Sunnyvale, California USA(2012).
42

Appendix A. Oblique Images of the Arrays
Figure A-1 Oblique Aerial View of the Canopy Array Site
A-1

Appendix A. Oblique Images of the Arrays
Figure A-2 Oblique Aerial View of the Ground Array Site
A-2

Appendix A. Oblique Images of the Arrays
Figure A-3 Oblique Aerial View of the Roof Array site
A-3

Appendix B. Manufacturer Data Sheets
Figure B-1 Manufacturer Datasheet for the PV Module at All Three Arrays
B-1

Appendix B. Manufacturer Data Sheets
Figure B-2 Manufacturer Datasheet for the Inverter at the Canopy and Ground Arrays
B-2

Appendix B. Manufacturer Data Sheets
Figure B-3 Manufacturer Datasheet for the Inverter at the Roof Array
B-3

Appendix C.  Copper Wire Gauges and Lengths
Table C-1  Copper Wire Gauges and Lengths
|          | Canopy  | Ground  | Roof  |
| -------- | ------- | ------- | ----- |
| Modules  |         |         |       |
Lead Cross-Sectional Area [mm2]
|                                            | 3.31  | 3.31  | 3.31  |
| ------------------------------------------ | ----- | ----- | ----- |
| Individual Lead Length [m]                 | 1.1   | 1.1   | 1.1   |
|                                            |       |       |       |
| Modules to Combiner Box                    |       |       |       |
| Conductor Cross-Sectional Area [mm2]       | 5.26  | 5.26  | 5.26  |
| Approximate Average Distance, One-Way [m]  | 85    | 95    | 52    |
|                                            |       |       |       |

| Combiner Box to Inverter DC Combiner  |     |     |     |
| ------------------------------------- | --- | --- | --- |
Compartment
| Conductor Cross-Sectional Area [mm2]       | 142*  | 127  | 53.5  |
| ------------------------------------------ | ----- | ---- | ----- |
| Approximate Average Distance, One-Way [m]  | 60    | 79   | 167   |
|                                            |       |      |       |
| Inverter DC Combiner Compartment to        |       |      |       |
Inverter Power Module Assembly
Compartment
| Conductor Cross-Sectional Area [mm2]        | 613  | 613  | 242   |
| ------------------------------------------- | ---- | ---- | ----- |
| Approximate Average Distance, One-Way [m]   | 4    | 4    | 4     |
|                                             |      |      |       |
| Inverter to Building Switchgear             |      |      |       |
| Conductor Cross-Sectional Area [mm2]        | 253  | 253  | 42.4  |
| Number of Wires per Phase                   | 1    | 2    | 1     |
| Total Conductor Cross-Sectional Area [mm2]  | 253  | 507  | 42.4  |
| Approximate Average Distance, One-Way [m]   | 172  | 79   | 14    |
* Weighted average of 127 mm2 and 177 mm2 conductors

C-1

Appendix D. Diagrams of the Arrays
Figure D-1 Diagram of the Canopy Array Strings and Pertinent Array Dimensions
D-1

Appendix D. Diagrams of the Arrays
Figure D-2 Diagram of the Ground Array Strings and Pertinent Array and Fence Dimensions
D-2

Appendix D. Diagrams of the Arrays
Figure D-3 Diagram of the Roof Array Strings and Pertinent Array Dimensions
D-3

Appendix E.  The Saved Data Values From Each Array
Table E-1  The Saved Data Values From Each Array
|                |                                 | One-Minute  | Canopy (C)/  |
| -------------- | ------------------------------- | ----------- | ------------ |
| Variable Name  | Units  Description              | Average/    | Ground (G)/  |
|                |                                 | Min/Max     | Roof (R)     |
| AmbTemp_C      | C  Outdoor ambient temperature  | Average     | C/G/R        |
Battery_A  A  Current to DAS battery, neg. = discharging  Average  C/G/R
| Battery_V  | V  Charge voltage of DAS battery  | Min  | C/G/R  |
| ---------- | --------------------------------- | ---- | ------ |
Charging regulator power source
0 = none
| ChgSource  | -   | Min  | C/G/R  |
| ---------- | --- | ---- | ------ |
1 = solar terminals
2 = continuous terminals
State of charging regulator;
-1 = regulator fault
0 = no charge
| ChgState  | -  1 = current limited charging  | Min  | C/G/R  |
| --------- | -------------------------------- | ---- | ------ |
2 = cycle charging
3 = float charging
4 = battery test
Status of DAS battery;
| CkBatt  | -  0 = normal  | Max  | C/G/R  |
| ------- | -------------- | ---- | ------ |
1 = check battery
Temperature of the data logger front
| CR1000Temp_C  | C   | Average  | C/G/R  |
| ------------- | --- | -------- | ------ |
terminal panel
Door sensor for DAS enclosure;
| DoorOpen  | -   | Min  | C/G/R  |
| --------- | --- | ---- | ------ |
-1 = open, 0 =closed
Inverter digital power control board fault
| DPCBFaults  | -   | Max  | R   |
| ----------- | --- | ---- | --- |
code
GeneralFaults1  -  Inverter general fault code, group 1  Max  R
GeneralFaults2  -  Inverter general fault code, group 2  Max  R
| HardwareFaults  | -  Inverter hardware fault code  | Max  | R    |
| --------------- | -------------------------------- | ---- | ---- |
| InvDriveFault   | -  Inverter drive fault code     | Max  | C/G  |
| InverterFaults  | -  Inverter inverter fault code  | Max  | R    |
AC total real energy from current day,
| InvEtoday_kWh  | kWh  | Max  | R   |
| -------------- | ---- | ---- | --- |
meas. by inverter
AC total accumulated real energy, meas. by
| InvEtot_kWh  | kWh  | Max  | C/G/R  |
| ------------ | ---- | ---- | ------ |
inverter
AC total accumulated real energy, meas. by
| InvEtot_kWh  | kWh  | Max  | R   |
| ------------ | ---- | ---- | --- |
inverter
AC total real energy from previous day,
| InvEyesterday_kWh  | kWh  | Max  | R   |
| ------------------ | ---- | ---- | --- |
meas. by inverter
InvFreq  Hz  AC frequency, meas. by inverter  Average  C/G/R
| InvGndImped_kOhm  | kΩ  Inverter ground impedance  | Average  | R   |
| ----------------- | ------------------------------ | -------- | --- |
E-1

| InvGridFault  | -  Inverter grid fault code  | Max  | C/G  |
| ------------- | ---------------------------- | ---- | ---- |
InvIa  A  Line current, phase A, meas. by inverter  Average  C/G/R
InvIabcAvg  A  Average AC line current, meas. by inverter  Average  R
InvIb  A  Line current, phase B, meas. by inverter  Average  C/G/R
InvIc  A  Line current, phase C, meas. by inverter  Average  C/G/R
InvIDCin  A  Inverter DC input current, meas. by inverter  Average  C/G/R
| InvIGnd  | A  Inverter DC ground current  | Average  | R   |
| -------- | ------------------------------ | -------- | --- |
InvINeutral  A  AC neutral current, meas. by inverter  Average  R
AC line current unbalance, meas. by
| InvIUnbal  | %   | Average  | R   |
| ---------- | --- | -------- | --- |
inverter
| InvMainFault  | -  Inverter main fault code         | Max      | C/G  |
| ------------- | ----------------------------------- | -------- | ---- |
| InvOpState    | -  Inverter operating status/state  | Average  | R    |
InvOpStatus  -  Inverter operating status/state  Average  C/G
InvPAC_kW  kW  AC real power, meas. by inverter  Average  C/G/R
InvPDC_kW  kW  Inverter DC input power, meas. by inverter  Average  C/G/R
| InvPF  | -  AC power factor, meas. by inverter  | Average  | R   |
| ------ | -------------------------------------- | -------- | --- |
InvPVMStatus  -  Inverter PV monitoring status code  Max  C/G
| InvSystemFault  | -  Inverter system fault code       | Max  | C/G  |
| --------------- | ----------------------------------- | ---- | ---- |
| InvSystemWarn   | -  Inverter system warning code     | Max  | C/G  |
| InvTempFault    | -  Inverter temperature fault code  | Max  | C/G  |
InvTempHeatsink_C  C  Inverter heatsink temperature  Average  R
Ambient temperature inside inverter general
| InvTempInternalAir_C  | C   | Average  | R   |
| --------------------- | --- | -------- | --- |
enclosure
Ambient temperature inside inverter
| InvTempInverterAir_C  | C   | Average  | R   |
| --------------------- | --- | -------- | --- |
inverter enclosure
InvVa  V  Line voltage, phase A, meas. by inverter  Average  C/G/R
InvVA_kVA  kVA  AC apparent power, meas. by inverter  Average  R
InvVabcAvg  V  Average AC line voltage, meas. by inverter  Average  R
InvVAR_kVAR  kVAR  AC reactive power, meas. by inverter  Average  R
InvVb  V  Line voltage, phase B, meas. by inverter  Average  C/G/R
InvVc  V  Line voltage, phase C, meas. by inverter  Average  C/G/R
Inverter DC input voltage on load side of
| InvVDCin  | V   | Average  | C/G/R  |
| --------- | --- | -------- | ------ |
DC contactor, meas. by inverter
Array DC voltage on DC busbars, measured
| InvVDVoltage_V  | V   | Average  | C/G/R  |
| --------------- | --- | -------- | ------ |
using voltage divider
| InvVoltageFault  | -  Inverter voltage fault code  | Max  | C/G  |
| ---------------- | ------------------------------- | ---- | ---- |
Array DC voltage on line (PV) side of DC
| InvVPVin  | V   | Average  | C/G/R  |
| --------- | --- | -------- | ------ |
contactor, meas. by inverter
E-2

AC line voltage unbalance, meas. by
| InvVUnbal  | %   | Average  | R   |
| ---------- | --- | -------- | --- |
inverter
Current supplied to load (DAS) from
| Load_A  | A   | Average  | C/G/R  |
| ------- | --- | -------- | ------ |
charging regulator
NumActiveFaults  -  Inverter number of active faults  Max  R
AC total accumulated reactive energy to
| PwrMtrEdel_kVARh  | kVARh  | Max  | C/G/R  |
| ----------------- | ------ | ---- | ------ |
(used by) inverter, meas. by AC meter
AC total accumulated real energy to (used
| PwrMtrEdel_kWh  | kWh  | Max  | C/G/R  |
| --------------- | ---- | ---- | ------ |
by) inverter, meas. by AC meter
AC total accumulated reactive energy from
| PwrMtrErec_kVARh  | kVARh  | Max  | C/G/R  |
| ----------------- | ------ | ---- | ------ |
(generated by) inverter, meas. by AC meter
AC total accumulated real energy from
| PwrMtrErec_kWh  | kWh  | Max  | C/G/R  |
| --------------- | ---- | ---- | ------ |
(generated by) inverter, meas. by AC meter
PwrMtrFreq  Hz  AC frequency, meas. by AC meter  Average  C/G/R
PwrMtrFreqMax  Hz  Max. AC frequency, meas. by AC meter  Max  C/G/R
PwrMtrFreqMin  Hz  Min. AC frequency, meas. by AC meter  Min  C/G/R
PwrMtrIa  A  Line current, phase A, meas. by AC meter  Average  C/G/R
PwrMtrIb  A  Line current, phase B, meas. by AC meter  Average  C/G/R
PwrMtrIc  A  Line current, phase C, meas. by AC meter  Average  C/G/R
PwrMtrP_kVA  kVA  AC apparent power, meas. by AC meter  Average  C/G/R
PwrMtrP_kVAR  kVAR  AC reactive power, meas. by AC meter  Average  C/G/R
PwrMtrP_kW  kW  AC real power, meas. by AC meter  Average  C/G/R
PwrMtrPF  -  AC power factor, meas. by AC meter  Average  C/G/R
PwrMtrPhaseRev  Hz  AC phase reversal, meas. by AC meter  Average  C/G/R
PwrMtrVa  V  Line voltage, phase A, meas. by AC meter  Average  C/G/R
AC max. total harmonic distortion in phase
| PwrMtrVaTHDMax  | %   | Max  | C/G/R  |
| --------------- | --- | ---- | ------ |
A, meas. by AC meter
PwrMtrVb  V  Line voltage, phase B, meas. by AC meter  Average  C/G/R
AC max. total harmonic distortion in phase
| PwrMtrVbTHDMax  | %   | Max  | C/G/R  |
| --------------- | --- | ---- | ------ |
B, meas. by AC meter
PwrMtrVc  V  Line voltage, phase C, meas. by AC meter  Average  C/G/R
AC max. total harmonic distortion in phase
| PwrMtrVcTHDMax  | %   | Max  | C/G/R  |
| --------------- | --- | ---- | ------ |
C, meas. by AC meter
Millivolt output from pyranometer;
Canopy: POA, west-tilted
| Pyra1_mV  | mV  | Average  | C/G/R  |
| --------- | --- | -------- | ------ |
Ground:  GHI
Roof:  GHI, southwest
Millivolt output from pyranometer;
Canopy: GHI
| Pyra2_mV  | mV  | Average  | C/G/R  |
| --------- | --- | -------- | ------ |
Ground:  POA
Roof:  GHI, southeast
E-3

Millivolt output from pyranometer;
Pyra3_mV mV Canopy: POA, east-tilted Average C/R
Roof: POA, southeast
Millivolt output from pyranometer;
Pyra4_mV mV Average R
Roof: GHI, northwest
Millivolt output from pyranometer;
Pyra5_mV mV Average R
Roof: GHI, northeast
Millivolt output from pyranometer;
Pyra6_mV mV Average R
Roof: GHI, center of older horiz. array
Millivolt output from pyranometer;
Pyra7_mV mV Roof: GHI, northwest of older horiz. Average R
array
Millivolt output from pyranometer;
Pyra8_mV mV Roof: GHI, northeast of older horiz. Average R
array
Qloss_Ah A·h Charge deficit of DAS battery Max C/G/R
Irradiance from flat-plate silicon sensor;
Canopy: POA, west-tilted
RefCell1_Wm2 W/m2 Average C/G/R
Ground: POA
Roof: POA, southeast
Irradiance from flat-plate silicon sensor;
Canopy: POA, east-tilted
RefCell2_Wm2 W/m2 Average C/R
Roof: GHI, center of older horiz. array,
east, cleaned
Irradiance from flat-plate silicon sensor;
RefCell3_Wm2 W/m2 Roof: GHI, center of older horiz. array, Average R
west, uncleaned
Temperature of module backsheet;
Canopy: east-tilted southwest module
RTD_C(1) C Ground: center shed, northwest module Average C/G/R
Roof: south sub-array, south-center
module
RTD_C(10) C Ambient temperature inside inverter Average C/G/
Temperature of module backsheet;
Canopy: east-tilted southeast module
Ground: center shed, center module of
RTD_C(2) C Average C/G/R
west half
Roof: center-east sub array, center-east
module
Temperature of module backsheet;
Canopy: west-tilted center-east module
Ground: center shed, center-bottom
RTD_C(3) C Average C/G/R
module
Roof: center-east sub array, center
module, center
E-4

Temperature of module backsheet;
Canopy: east-tilted center-west module,
center
RTD_C(4) C Average C/G/R
Ground: center shed, center module, center
Roof: center-east sub array, center
module, northwest
Temperature of module backsheet;
Canopy: east-tilted center-west module,
southwest
RTD_C(5) C Ground: center shed, center module, Average C/G/R
southwest
Roof: center-east sub array, center
module, south-center
Temperature of module backsheet;
Canopy: east-tilted center-west module,
center-east
RTD_C(6) C Ground: center shed, center module, Average C/G/R
center-east
Roof: center-east sub array, center
module, center-east
Temperature of module backsheet;
Canopy: east-tilted center-west module,
north-center
RTD_C(7) C Ground: center shed, center module, north- Average C/G/R
center
Roof: northwest sub-array, northwest
module
Temperature of module backsheet;
RTD_C(8) C Canopy: east-tilted center-east module Average C/G/
Ground: center shed, center-top module
Temperature of module backsheet;
RTD_C(9) C Canopy: east-tilted northwest module Average C/G/
Ground: north shed, center module
Outdoor ambient temperature from
SEWSAmbientTemp_C C Average C/G/R
integrator’s sensor
Module temperature from integrator’s
SEWSModuleTemp_C C Average C/G/R
sensor
POA irradiance from integrator’s domed
SEWSPOAIrrad_Wm2 W/m2 Average C/G/R
diffused silicon-cell pyranometer
ShuntCurrent_A(1) A Output circuit current from combiner box 1 Average C/G/R
ShuntCurrent_A(2) A Output circuit current from combiner box 2 Average C/G/R
ShuntCurrent_A(3) A Output circuit current from combiner box 3 Average C/G/R
ShuntCurrent_A(4) A Output circuit current from combiner box 4 Average C/G/R
ShuntCurrent_A(5) A Output circuit current from combiner box 5 Average C/G
ShuntCurrent_A(6) A Output circuit current from combiner box 6 Average C/G
ShuntCurrent_A(7) A Output circuit current from combiner box 7 Average C/G
E-5

Output circuit total accumulated energy
ShuntEtot_kWh_Avg(1) kWh from combiner box 1, meas. using shunts - C/G/R
and voltage divider*
Output circuit total accumulated energy
ShuntEtot_kWh_Avg(2) kWh from combiner box 2, meas. using shunts - C/G/R
and voltage divider*
Output circuit total accumulated energy
ShuntEtot_kWh_Avg(3) kWh from combiner box 3, meas. using shunts - C/G/R
and voltage divider*
Output circuit total accumulated energy
ShuntEtot_kWh_Avg(4) kWh from combiner box 4, meas. using shunts - C/G/R
and voltage divider*
Output circuit total accumulated energy
ShuntEtot_kWh_Avg(5) kWh from combiner box 5, meas. using shunts - C/G
and voltage divider*
Output circuit total accumulated energy
ShuntEtot_kWh_Avg(6) kWh from combiner box 6, meas. using shunts - C/G
and voltage divider*
Output circuit total accumulated energy
ShuntEtot_kWh_Avg(7) kWh from combiner box 7, meas. using shunts - C/G
and voltage divider*
Output circuit power from combiner box 1,
ShuntPDC_kW_Avg(1) kW - C/G/R
meas. using shunts and voltage divider*
Output circuit power from combiner box 2,
ShuntPDC_kW_Avg(2) kW - C/G/R
meas. using shunts and voltage divider*
Output circuit power from combiner box 3,
ShuntPDC_kW_Avg(3) kW - C/G/R
meas. using shunts and voltage divider*
Output circuit power from combiner box 4,
ShuntPDC_kW_Avg(4) kW - C/G/R
meas. using shunts and voltage divider*
Output circuit power from combiner box 5,
ShuntPDC_kW_Avg(5) kW - C/G
meas. using shunts and voltage divider*
Output circuit power from combiner box 6,
ShuntPDC_kW_Avg(6) kW - C/G
meas. using shunts and voltage divider*
Output circuit power from combiner box 7,
ShuntPDC_kW_Avg(7) kW - C/G
meas. using shunts and voltage divider*
TemperatureFaults - Inverter temperature fault code Max R
TIMESTAMP LST Date and time Max C/G/R
Warnings - Inverter system warning code Max R
WindDir_deg ° Wind direction, clockwise from north - C/G/R
WindDirAve_deg ° Unit vector mean wind direction* - C/G/R
WindDirStdDev_deg ° Standard deviation of the wind direction* - C/G/R
E-6

1 = heating disabled
2 = enabled, not heating
| WindHeatStateID  | -  3 = 50 % heating  | Average  | C/G/R  |
| ---------------- | -------------------- | -------- | ------ |
4 = 100 % heating
5 = error
WindRef_V  V  Internal reference voltage in wind sensor  Min  C/G/R
| WindSpeed_ms     | m/s  Wind speed                   | Max  | C/G/R  |
| ---------------- | --------------------------------- | ---- | ------ |
| WindSpeedAve_ms  | m/s  Mean horizontal wind speed*  | -    | C/G/R  |
Wind sensor data validity
| WindValid  | -  -1 = wind data is valid  | Average  | C/G/R  |
| ---------- | --------------------------- | -------- | ------ |
0 = wind data is invalid
Note: All measurements are sampled every 1 s except the RTDs, which are sampled every 10 seconds. All
measurements are saved every 1 s, except those marked with a “*”.
* Calculated from saved 1 s samples; only the 1 min aggregate data are saved
|     |     |     |     |
| --- | --- | --- | --- |
E-7

Appendix F. Anonymized Diagram of the Data Acquisition Systems’ Communications Network
Figure F-1 Anonymized Diagram of the Data Acquisition Systems’ Communications Network
F-1