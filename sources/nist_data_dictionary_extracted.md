Data Dictionary for NIST Photovoltaic Array and Weather
Station Datasets, v1.0
Last update: July 27, 2017

Part 1.  PV Array Datasets
The Saved Data Values from Each Array
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
| InverterFaults  | -  Inverter fault code           | Max  | R    |
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
enclosure
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

Part 2.  Weather Station Dataset
The Saved Data Values from the Weather Station
Top (T)/
One-Minute
| Variable Name  | Units  Description  |     | Lower (L)  |
| -------------- | ------------------- | --- | ---------- |
Aggregation
Data Logger
| AirMass      | -  Absolute air mass       | Average  | T/L  |
| ------------ | -------------------------- | -------- | ---- |
| AirPres_kPa  | kPa  Atmospheric pressure  | Average  | T    |
Outdoor ambient temperature 5.59 m above
| AirTemp_C  | C   | Average  | T   |
| ---------- | --- | -------- | --- |
roof
AM25TRefTemp_C  C  Thermocouple UTR reference temperature  Average  L
Outdoor ambient temperature 1.07 m above
| AmbTemp_C  | C   | Average  | L   |
| ---------- | --- | -------- | --- |
roof
Mean ambient temperature ventilator fan
| AmbVent_rpm  | rev/min  | Average*  | L   |
| ------------ | -------- | --------- | --- |
speed
Battery_A  A  Current to UPS battery, neg. = discharging  Average  T/L
| Battery_V  | V  Charge voltage of UPS battery  | Min  | T/L  |
| ---------- | --------------------------------- | ---- | ---- |
Charging regulator power source
0 = none
| ChgSource  | -   | Min  | T/L  |
| ---------- | --- | ---- | ---- |
1 = solar terminals
2 = continuous terminals
State of charging regulator;
-1 = regulator fault
0 = no charge
| ChgState  | -  1 = current limited charging  | Min  | T/L  |
| --------- | -------------------------------- | ---- | ---- |
2 = cycle charging
3 = float charging
4 = battery test
Status of UPS battery;
| CkBatt  | -  0 = normal  | Max  | T/L  |
| ------- | -------------- | ---- | ---- |
1 = check battery
Temperature of the data logger front
| CR1000Temp_C  | C   | Average  | T/L  |
| ------------- | --- | -------- | ---- |
terminal panel
| Declination_deg  | °  Earth’s declination angle  | Average  | T/L  |
| ---------------- | ----------------------------- | -------- | ---- |
Door sensor for DAS enclosure;
| DoorOpen  | -   | Min  | T   |
| --------- | --- | ---- | --- |
-1 = open, 0 =closed
| Hail_hitssqrcm  | hits/cm2  Hail count  | Sum  | T   |
| --------------- | --------------------- | ---- | --- |
Current supplied to load (UPS) from
| Load_A  | A   | Average  | T/L  |
| ------- | --- | -------- | ---- |
charging regulator
PwrStripState  -  State of remote power strip, bit field  Min  L
Millivolt output from thermopile
| Pyra1_mV  | mV  | Average  | T   |
| --------- | --- | -------- | --- |
pyranometer, south table position
Thermopile pyranometer case temperature,
| Pyra1temp_C  | C   | Average  | T   |
| ------------ | --- | -------- | --- |
south table position

Millivolt output from thermopile
| Pyra2_mV  | mV  | Average  | L   |
| --------- | --- | -------- | --- |
pyranometer, north table position
Thermopile pyranometer case temperature,
| Pyra2temp_C  | C   | Average  | L   |
| ------------ | --- | -------- | --- |
north table position
Millivolt output from diffuse-measuring
Pyrad1_mV  mV  pyranometer, east tracker position (when  Average  T
pointing south)
Millivolt output from diffuse-measuring
Pyrad2_mV  mV  pyranometer, west tracker position (when  Average  L
pointing south)
Millivolt output from domed-diffused
| PyraS1_Wm2  | W/m2  | Average  | L   |
| ----------- | ----- | -------- | --- |
silicon-cell pyranometer
| Pyrg1_mV     | mV  Millivolt output from pyrgeometer  | Average  | T   |
| ------------ | -------------------------------------- | -------- | --- |
| Pyrg1temp_C  | C  Pyrgeometer case temperature        | Average  | T   |
Millivolt output from pyrheliometer, east
| Pyrh1_mV  | mV  | Average  | T   |
| --------- | --- | -------- | --- |
tracker position (when pointing south)
Pyrheliometer case temperature, east tracker
| Pyrh1temp_C  | C   | Average  | T   |
| ------------ | --- | -------- | --- |
position (when pointing south)
Millivolt output from pyrheliometer, west
| Pyrh2_mV  | mV  | Average  | L   |
| --------- | --- | -------- | --- |
tracker position (when pointing south)
| Qloss_Ah  | A·h  Charge deficit of UPS battery  | Max  | T/L  |
| --------- | ----------------------------------- | ---- | ---- |
Rain_mm  mm  Rain accumulation, piezoelectric sensor  Sum  T
Rain and precipitation liquid equivalent
| RainTB_mm  | mm  | Sum  | T   |
| ---------- | --- | ---- | --- |
accumulation, tipping bucket gauge
Irradiance from flat-plate silicon sensor, 0°
| RefCell1_Wm2  | W/m2  | Average  | T   |
| ------------- | ----- | -------- | --- |
tilt
Irradiance from flat-plate silicon sensor, 10°
W/m2
| RefCell2_Wm2  |     | Average  | T   |
| ------------- | --- | -------- | --- |
tilt, 180° azimuth CW from north
Irradiance from flat-plate silicon sensor, 5°
| RefCell3_Wm2  | W/m2  | Average  | T   |
| ------------- | ----- | -------- | --- |
tilt, 270° azimuth CW from north
Irradiance from flat-plate silicon sensor, 5°
| RefCell4_Wm2  | W/m2  | Average  | T   |
| ------------- | ----- | -------- | --- |
tilt, 90° azimuth CW from north
Irradiance from flat-plate silicon sensor, 20°
| RefCell5_Wm2  | W/m2  | Average  | T   |
| ------------- | ----- | -------- | --- |
tilt, 180° azimuth CW from north
W/m2  Irradiance from flat-plate silicon sensor,
| RefCell6_Wm2  |     | Average  | T   |
| ------------- | --- | -------- | --- |
18.4° tilt, 180° azimuth CW from north
| RelayState  | -  State of latching relay, bit field  | Min      | T/L  |
| ----------- | -------------------------------------- | -------- | ---- |
| RelHumid    | %  Relative humidity                   | Average  | T    |
Temperature of module backsheet: 10°
| RTD_C(1)  | C   | Average  | L   |
| --------- | --- | -------- | --- |
tilted module, center
Temperature of module backsheet: 5° east
| RTD_C(10)  | C   | Average  | L   |
| ---------- | --- | -------- | --- |
tilted module, southwest
Temperature of module backsheet: 5° east
| RTD_C(11)  | C   | Average  | L   |
| ---------- | --- | -------- | --- |
tilted module, center-east

Temperature of module backsheet: 5° east
| RTD_C(12)  | C   | Average  | L   |
| ---------- | --- | -------- | --- |
tilted module, north-center
Temperature of module backsheet: 20°
| RTD_C(13)  | C   | Average  | L   |
| ---------- | --- | -------- | --- |
tilted module, center
Temperature of module backsheet: 20°
| RTD_C(14)  | C   | Average  | L   |
| ---------- | --- | -------- | --- |
tilted module, northeast
Temperature of module backsheet: 20°
| RTD_C(15)  | C   | Average  | L   |
| ---------- | --- | -------- | --- |
tilted module, center-west
Temperature of module backsheet: 20°
| RTD_C(16)  | C   | Average  | L   |
| ---------- | --- | -------- | --- |
tilted module, south-center
Temperature of module backsheet: 10°
| RTD_C(2)  | C   | Average  | L   |
| --------- | --- | -------- | --- |
tilted module, northwest
Temperature of module backsheet: 10°
| RTD_C(3)  | C   | Average  | L   |
| --------- | --- | -------- | --- |
tilted module, south-center
Temperature of module backsheet: 10°
| RTD_C(4)  | C   | Average  | L   |
| --------- | --- | -------- | --- |
tilted module, center-east
Temperature of module backsheet: 5° west
| RTD_C(5)  | C   | Average  | L   |
| --------- | --- | -------- | --- |
tilted module, center
Temperature of module backsheet: 5° west
| RTD_C(6)  | C   | Average  | L   |
| --------- | --- | -------- | --- |
tilted module, northeast
Temperature of module backsheet: 5° west
| RTD_C(7)  | C   | Average  | L   |
| --------- | --- | -------- | --- |
tilted module, center-west
Temperature of module backsheet: 5° west
| RTD_C(8)  | C   | Average  | L   |
| --------- | --- | -------- | --- |
tilted module, south-center
Temperature of module backsheet: 5° east
| RTD_C(9)  | C   | Average  | L   |
| --------- | --- | -------- | --- |
tilted module, center
Weather transmitter heating state:
1 = heating disabled
2 = enabled, not heating
| SixInOneHeatStateID  | -   | Average  | T   |
| -------------------- | --- | -------- | --- |
3 = 50 % heating
4 = 100 % heating
5 = error
Internal reference voltage in weather
| SixInOneRef_V  | V   | Min  | T   |
| -------------- | --- | ---- | --- |
transmitter
| SnowDepth_cm  | cm  Snow depth  | Average  | L   |
| ------------- | --------------- | -------- | --- |
SolarAzFromSouth_deg  °  Solar azimuth, CW from south  Average  T/L
| SolarTime_hr     | h  Solar time          | Max, Median  | T/L  |
| ---------------- | ---------------------- | ------------ | ---- |
| SolarZenith_deg  | °  Solar zenith angle  | Average      | T/L  |
Temperature of module backsheet: 0° tilted
| TC_C(1)  | C   | Average  | L   |
| -------- | --- | -------- | --- |
module, center
Temperature of module backsheet: 0° tilted
| TC_C(2)  | C   | Average  | L   |
| -------- | --- | -------- | --- |
module, southeast
Temperature of module backsheet: 0° tilted
| TC_C(3)  | C   | Average  | L   |
| -------- | --- | -------- | --- |
module, north-center

Temperature of module backsheet: 0° tilted
| TC_C(4)  | C   | Average  | L   |
| -------- | --- | -------- | --- |
module, between center and southeast
Temperature of module backsheet: 18.4°
| TC_C(5)  | C   | Average  | L   |
| -------- | --- | -------- | --- |
tilted module, center
Temperature of module backsheet: 18.4°
| TC_C(6)  | C   | Average  | L   |
| -------- | --- | -------- | --- |
tilted module, northwest
Temperature of module backsheet: 18.4°
| TC_C(7)  | C   | Average  | L   |
| -------- | --- | -------- | --- |
tilted module, south-center
Temperature of module backsheet: 18.4°
| TC_C(8)  | C   | Average  | L   |
| -------- | --- | -------- | --- |
tilted module, center-east
| TIMESTAMP     | LST  Date and time            | Max  | T/L  |
| ------------- | ----------------------------- | ---- | ---- |
| TrackerFlags  | -  Tracker status, bit field  | Max  | L    |
| TrackerState  | -  Tracker state              | Min  | L    |
Millivolt output from UV-A radiometer
| UVA_mV  | mV  | Average  | T   |
| ------- | --- | -------- | --- |
converter box
Millivolt output from UV-B radiometer
| UVB_mV  | mV  | Average  | T   |
| ------- | --- | -------- | --- |
converter box
UV-B radiometer optical system
| UVBtemp_C  | C   | Average  | T   |
| ---------- | --- | -------- | --- |
temperature
UVT_mV  mV  Millivolt output from UV total radiometer  Average  L
Mean GHI ventilator fan speed, south table
| Vent2_rpm  | rev/min  | Average*  | T   |
| ---------- | -------- | --------- | --- |
position
Mean GHI ventilator fan speed, north table
| Vent4_rpm  | rev/min  | Average*  | L   |
| ---------- | -------- | --------- | --- |
position
Wind direction 5.72 m above roof,
| WindDir_deg  | °   | -   | T   |
| ------------ | --- | --- | --- |
clockwise from north
Wind direction 2.51 m above roof,
| WindDir1_deg  | °   | -   | L   |
| ------------- | --- | --- | --- |
clockwise from north
Unit vector mean wind direction 2.51 m
| WindDir1Ave_deg  | °   | Average*  | L   |
| ---------------- | --- | --------- | --- |
above roof
Unit vector mean wind direction 5.72 m
| WindDirAve_deg  | °   | Average*  | T   |
| --------------- | --- | --------- | --- |
above roof
Standard deviation of the wind direction;
WindDirStdDev_deg  °  Top:  5.72 m above roof  Std. Dev.*  T/L
Lower:  2.51 m above roof
| WindSpeed_ms   | m/s  Wind speed 5.72 m above roof  | Max  | T   |
| -------------- | ---------------------------------- | ---- | --- |
| WindSpeed1_ms  | m/s  Wind speed 2.51 m above roof  | Max  | L   |
Mean horizontal wind speed 2.51 m above
| WindSpeed1Ave_ms  | m/s  | Average*  | L   |
| ----------------- | ---- | --------- | --- |
roof
Mean horizontal wind speed 5.72 m above
| WindSpeedAve_ms  | m/s  | Average*  | T   |
| ---------------- | ---- | --------- | --- |
roof

Wind sensor 5.72 m above roof data
validity
WindValid - Average T
-1 = wind data is valid
0 = wind data is invalid
Note: All measurements are sampled every 1 s except the RTDs and thermocouples, which are sampled every 10 s.
All measurements are saved every 1 s, except those marked with a “*”.
* Calculated from saved 1 s samples; only the 1 min aggregate data are saved