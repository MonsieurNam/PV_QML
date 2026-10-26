# NIST Roof metadata verification

Retrieved 2026-10-06. Scope: the Roof array, one-minute aggregates, 2015–2016.
No model fitting, validation tuning, holdout scoring or human approval occurred.
Research-lookup's API backends were unavailable (no keys); direct official
sources and public portal metadata were used. MarkItDown generated the text
extracts; Excel comments were inspected separately because conversion omits them.

## Timestamp semantics — confirmed

[NIST Technical Note 1896](https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.1896.pdf),
section 4.1.5.7, printed page 36, specifies that the minute's stored aggregate
uses samples from the preceding 60 seconds. This supports interval-end labels.
The report also documents the Roof logger's scheduled resets; these are not
equivalent to a dated outage inventory.

The official Instrument History workbook additionally identifies its recorded
installation/removal timestamps as good one-minute observations incorporating
prior one-second values. Roof cells C1/D1 specify EST and warn that 12:00:01
installation/removal times can be placeholders rather than precise event times.

[Dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7339723/) confirms
fixed EST without daylight saving and describes an issues log. The current
[testbed page](https://www.nist.gov/el/beed/heat-transfer-alternative-energy-systems/photovoltaic-testbeds)
links the [portal](https://pvdata.nist.gov/) and
[data dictionary](https://www.nist.gov/file/391591).

## Available instrument history — reviewed, partial coverage

The public portal's Box storage path for the downloaded 2015 Roof ZIP leads to
`pvdata` folder 72548100973. Its `Documents` folder 83269133057 contains one
file: `Instrument_History.xlsx`, Box file ID 499249265751. This provenance was
obtained with the same public portal endpoint, not a personal Box account.
Temporary access tokens and signed download URLs were not saved.

The workbook contains five visible sheets. `Roof` has 32 rows and eight columns.
It records sensor installation/removal, model, serial number, calibration and
comments. The chosen RefCell2 sensor spans the study period; the outdoor
temperature sensor also covers it. Documented changes to other pyranometers or
wind equipment must not invalidate our four channels wholesale. Inverter trips
and clipping remain physical events, not automatically discarded sensor errors.

This workbook does **not** supply a complete dated outage/erroneous-reading
inventory for the AC power meter and all selected channels. It must not be
presented as proof that there were no other faults. The separate Net-Zero house
log is a different facility and is not substituted. Completeness of the Roof
issues log remains unresolved; primary preparation/G0 is still HOLD pending
adequate source confirmation or an explicitly approved protocol amendment.

## Downloaded immutable originals

All originals are under `artifacts/nist/raw/` in the D-drive project. Derived
Markdown extracts are under `sources/` and do not replace the originals.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| NIST.TN.1896.pdf | 7180738 | fe434d70f0dcd77844751c76c2f756b78ecc5ccdf76fffc0247678c2d4211a2b |
| jres.122.040.pdf | 99768 | 7d18692a7b128cd8c52a8846cb4166c311c9fc4c54fe541f92e58c83a30150df |
| data-dictionary.pdf | 232151 | e18eed11be12b085e34637105dec6dd70120aaf1a7d05d8caa8d256e43900bef |
| Instrument_History.xlsx | 44140 | 00fbae89f2535b12efaa1a07d17ccff18483d2e5f1c34b141e5a1e73db16e1bc |

Official dataset identifier: [10.18434/M3S67G](https://doi.org/10.18434/M3S67G).
The [government catalog](https://catalog.data.gov/dataset/nist-campus-photovoltaic-pv-arrays-and-weather-station-data-sets)
lists William M. Healy as the dataset contact. No email has been sent; downloading
and researching sources does not authorize correspondence on the user's behalf.
