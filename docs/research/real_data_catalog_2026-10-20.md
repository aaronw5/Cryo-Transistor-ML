# Real multi-device MOSFET I-V data catalog, 2026-10-20 (second pass)

Scope: public measured MOSFET I-V data (Id-Vgs and Id-Vds sweeps) with several
devices from one process, raw numeric files, and an open compact-model card or
PDK for a starting card. Multiple temperatures are a bonus.

This pass does not repeat the sources assessed in the I3-1 catalog
(`out/research/explore/I3_1/real_data_catalog.json`, commit 3fdebd0): TU Delft
40 nm (4TU), IHP SG13G2 Open PDK MOS measurements, UTA pyvisa_instrument_control,
Kabaoglu BSIM3 repository, SUSTech IEEE DataPort set, and the negative
findings listed there. Two I3-1 entries are re-assessed below because the
criteria changed (I3-1 required at least 4 temperatures; this pass requires
several devices): NIST mds2-2997, which I3-1 could not download, and
`google/skywater-pdk-sky130-raw-data`, which I3-1 rejected as single
temperature. The GF180MCU regression data (I3-1 rank 7) is re-checked because
I3-1 left open whether it is measured.

Verification labels:

- `listed`: file names and sizes read from a ZIP central directory (HTTP range
  request), a Git tree listing, or a repository API; no bulk download.
- `read`: one or a few small files (at most 3.5 MB each) fetched into a new
  empty directory under the session scratchpad (`.../scratchpad/datasearch/`)
  and read with `python3 -I` scripts kept in a separate directory, or with
  plain text tools.
- `page`: taken from a landing page or metadata API only.

## Table

| # | Source and URL | License | Process | Devices and geometries | Temperatures | Bias ranges | Format and size | Open card | Download check |
|---|---|---|---|---|---|---|---|---|---|
| 1 | NIST mds2-2997, "CMOS Sky130 Primitives measured at cryogenic temperatures", ETest Tile. https://doi.org/10.18434/mds2-2997, file `https://data.nist.gov/od/ds/mds2-2997/ETest%20Tile.zip` | NIST public data statement (https://www.nist.gov/open/license) | SkyWater sky130 (SkyWater ETest die) | 114 devices, each with one Id-Vg and one Id-Vd file. nfet_01v8 5, nfet_01v8_lvt 5, pfet_01v8 5, pfet_01v8_lvt 5, pfet_01v8_hvt 5 (W 0.42-7 um, L 0.15-8 um), nfet_03v3_nvt 3, nfet_05v0_nvt 5, nfet_g5v0d10v5 46 (40 W/L), pfet_g5v0d10v5 35 (34 W/L). File names carry W, L, m and module/pad numbers. | 4 K (chuck temperature, from README and file suffix `_4K`; no TEMP field in the MDM header). Room-temperature files for the same 114 module/pad IDs are in source 3. | 1.8 V devices: Id-Vg Vgs 0-1.8 V in 10 mV (181 points) at Vds 0.1 and 1.8 V, Vbs 0, -0.75, -1.5 V; Id-Vd Vds 0-1.8 V in 25 mV, Vgs 0-1.8 V in 0.3 V steps, Vbs 0 and -1.5 V. Thick-oxide devices: same structure to 3.3 V (README). pMOS mirrored. | IC-CAP MDM text; 230 entries, 3,807,177 bytes zipped | sky130 BSIM4 cards (room temperature, binned) in the SkyWater PDK, Apache-2.0; no open 4 K card | HEAD 200 with Content-Length 3,807,177 on 2026-10-20 19:35 EDT. Central directory `listed`. README, one nfet_01v8 W=7/L=8 Id-Vg/Id-Vd pair `read`: 6 Id-Vg blocks of 181 points, Id(Vgs=Vds=1.8 V, Vbs=0) = 4.53e-4 A at 4 K vs 1.53e-4 A in the room-temperature file of the same module (source 3). |
| 2 | NIST mds2-2997, MPW-5 Test Tile. File `https://data.nist.gov/od/ds/mds2-2997/MPW-5%20Test%20Tile.zip` | as 1 | SkyWater sky130 (MPW-5 test tile from NIST/ADSR/U. Michigan; GDS at https://github.com/msaligane/openfasoc_cryo_caravel) | 287 transistors with both Id-Vg and Id-Vd files: nthin (nfet_01v8) 80, pthin (pfet_01v8) 71, nthin_lvt (nfet_01v8_lvt) 56, pthin_lvt (pfet_01v8_lvt) 40, nnative (3.3 V native nMOS) 40. Files are named `<module>_<position>_IDVG.mdm`; `Module Layout Coding.xlsx` maps position to W/L. From that sheet: nfet_01v8 80 W/L (W 0.36-7 um, L 0.15-20 um), pfet_01v8 72 W/L (W 0.42-7, L 0.15-20), nfet_01v8_lvt 56 W/L (W 0.42-7, L 0.15-8). The pfet_01v8_lvt and native mapping was not checked. Folder readmes: pthin `17_6 gate not connected`, nthin_lvt `23_4 Issue`. | 4 K for the transistor module. `RT-for reference only/` holds 5 room-temperature devices (one per type). Ring oscillators at 4, 20, 50, 100 K (not I-V). | nthin: Id-Vg Vgs 0-1.8 V in 10 mV (181 points) at Vds 0.1 and 1.8 V, Vbs 0, 0.3, -0.4, -0.8, -1.2 V (10 curves); Id-Vd Vds 0-1.8 V in 20 mV (91 points), Vgs 0-1.8 V in 0.3 V steps, Vbs 0 and -1 V (14 curves). pthin_lvt: mirrored, Vbs 0, -0.3, 0.4, 0.8, 1.2 V. nnative: Vgs 0-3.3 V in 10 mV (331 points), Vds 0.1 and 3.3 V. | IC-CAP MDM text plus pptx/xlsx layout documents; 909 entries, 39,471,626 bytes zipped | as 1 | HEAD 200 with Content-Length 39,471,626. Central directory `listed`. Tile README, layout xlsx (114,976 bytes), three MDM files `read`. |
| 3 | google/skywater-pdk-sky130-raw-data. https://github.com/google/skywater-pdk-sky130-raw-data | Apache-2.0 | SkyWater sky130 (SkyWater test tile) | 451 devices with both Id-Vg and Id-Vd files, 14 device types: nfet_01v8 45 (19 W/L), nfet_01v8_lvt 52 (15), pfet_01v8 49 (11), pfet_01v8_hvt 40 (11), pfet_01v8_lvt 29 (9), nfet_03v3_nvt 13, nfet_05v0_nvt 13, nfet_g5v0d10v5 66 (43), pfet_g5v0d10v5 58 (39), nfet_g5v0d16v0 45, pfet_g5v0d16v0 14, ESD devices 27. All 114 module/pad IDs of source 1 are present here with both sweeps. | Room temperature (not recorded in files; I3-1 found no TEMP field) | nfet_01v8 W=7/L=8 file: Id-Vg Vgs 0-1.8 V in 50 mV (37 points) at Vds 0.1 and 1.8 V, Vbs 0, -0.9, -1.8 V | IC-CAP MDM text; 1,380 MDM files, 50,252,651 bytes | as 1 | Git tree `listed` via GitHub API; one Id-Vg file (16,320 bytes) `read`. |
| 4 | IHP SG13G2 Open PDK, `ihp-sg13g2/libs.doc/meas/MOS` (already in use as the 28-device dataset). https://github.com/IHP-GmbH/IHP-Open-PDK | Apache-2.0 | IHP SG13G2 130 nm | 129 DUTs with dc_idvg/dc_idvd files at 300 K: LV nMOS 38, LV pMOS 38, HV nMOS 27, HV pMOS 26. LV W 0.15-10 um, L 0.12-10 um. Of these, 28 have 4 temperatures (the set in use); the other 101 have 300 K only. | 300 K for 129 DUTs; 233, 343, 398 K for 28 DUTs | as recorded in I3-1 | MDM; already downloaded for the 28-DUT subset | IHP SG13G2 PSP 103.6 cards in the same repository | Git tree `listed` on branches main and dev; no new MOS data on dev. |
| 5 | Zenodo 8009712, "Data for Cryogenic Characterization and Modeling of 14 nm Bulk FinFET Technology" (A. Chabane, IBM Research Zurich; article record 10.5281/zenodo.6901637). https://doi.org/10.5281/zenodo.8009712 | CC-BY-4.0 | 14 nm bulk FinFET (foundry not named) | About 7 devices: NMOS 8H, 64H, 8SL, 64SL, NMOS3, PMOS 8H, 64H (8 or 64 fins, two flavors), several chips | 44 mK, 0.4, 1, 2, 3, 4, 4.6, 10, 25, 50, 100, 150, 200, 250, 290-310 K, different subsets per device (from file names). Example: NMOS 8H chip 3 at 100/150/200/250/310 K; NMOS64SL chip 6 at 4/50/100/200/300 K; PMOS 8H at 4.6/10/25/100/200/300 K. | Vmax 0.8 V (2021 sets, 5 mV steps) and up to 1.2 V (Vds list 0.2-1.2 V) | 62 CSV files in one 7z, 1,113,314 bytes | No open card for this process; only generic BSIM-CMG cards | 7z (1.1 MB) downloaded to its own directory and `listed` with `tar -tvf`; no file content read. |
| 6 | GF180MCU fd_pr regression data, `models/ngspice/testing/180MCU_SPICE_DATA/MOS` (also copied under `regression/*/measured_data`). https://github.com/google/globalfoundries-pdk-libs-gf180mcu_fd_pr | Apache-2.0 | GF 180MCU | nmos_3p3_iv: 25 W/L (W 0.22-10 um, L 0.28-10 um) | 25, -40, 125 C (assigned by the regression script) | Vgs to 3.3 V, Vbs 0 to -3.3 V, Vgs 0.8-3.3 V for Id-Vd | xlsx, about 3.4-4.4 MB per device type | GF180MCU BSIM4 cards, Apache-2.0 | `nmos_3p3_iv.nl_out.xlsx` (3,443,614 bytes) `read`: corner column `typical`, shared string `\HSPICE R-2020.12-SP1` and `Time taken`, leakage floor 1.0170e-14 A repeated across devices. The file is an HSPICE model output, not silicon data, although the folder is named `measured_data`. |
| 7 | ESDED, "Emerging Semiconductor Device Electrical Dataset", IEEE DataPort, doi:10.21227/e4zb-x710 | not stated; IEEE DataPort subscription required | multiple, not stated | not stated | recorded per entry (Temp field) | not stated | JSON, two archives of 141.73 MB and 85.84 MB | not applicable | `page` only. Origin (measured, digitized, or simulated) not stated on the landing page or the XJTLU abstract page. |
| 8 | MESD, "MOSFET Electrical Simulation Dataset", IEEE DataPort, doi:10.21227/wa5a-r991 | IEEE DataPort | 11 anonymized PDKs, 3-350 nm | 44 device types | several | several | not checked | not applicable | `page` only. Simulated from foundry BSIM models; not measured. |

## Notes per source

1. NIST mds2-2997 (sources 1 and 2). The NIST file server answered on
   2026-10-20 at 19:35 EDT; the I3-1 attempts at 00:40-01:05 EDT the same day
   returned HTTP 524. The record is version 2.0.0 (README) with two parts
   measured on different dies: the ETest tile (March 2022) and the MPW-5 tile
   (November 2023). Both were measured in a Lakeshore closed-cycle probe
   station with an Agilent 4156B at a 4 K chuck temperature. MDM headers have
   no temperature field. The ETest README states that room-temperature
   measurements of the same tile are in `google/skywater-pdk-sky130-raw-data`;
   the module/pad IDs of all 114 ETest devices match files in that repository.
   The README does not state whether the room-temperature and 4 K files come
   from the same physical die. The ETest Id-Vg files also contain `beta` and
   `mu` output columns, which the README says to ignore.
2. The ETest 1.8 V core set has 5 devices per type (25 devices across
   nfet_01v8, nfet_01v8_lvt, pfet_01v8, pfet_01v8_lvt, pfet_01v8_hvt). Most of
   the 114 devices are the 5 V/10.5 V g5v0d10v5 type (81 devices, 74 W/L).
3. The MPW-5 transistor files do not carry W/L in the file name. The mapping
   is in `Module Layout Coding.xlsx`, sheet `FETModules` (column = module
   number, row = position 1-8, with pin assignment). Parasitics and exact
   layout sizes are in the GDS in `msaligane/openfasoc_cryo_caravel`.
4. Source 3 sweeps are coarser (50 mV) than the NIST 4 K sweeps (10 mV). The
   Vbs values also differ (0, -0.9, -1.8 V at room temperature vs 0, -0.75,
   -1.5 V at 4 K for ETest).
5. Source 4 is already used by the project. The 101 DUTs measured only at
   300 K are in the same repository and were not part of the I3-1 download.
6. Source 5 has fewer than 8 geometries and no open card for the process.
7. Source 6 resolves the I3-1 open question: the GF180MCU regression files are
   model outputs. No measured GF180MCU transistor I-V was found.

## Negative findings in this pass

| Source | Checked | Finding |
|---|---|---|
| UTA posters: A. Rimal, CPAD 2024 (indico.phy.ornl.gov/event/510/contributions/2205) and CPAD 2025 (indico.global/event/14966/contributions/133413); F. Beall, EDIT 2026 poster (indico.cern.ch/event/1472419) | landing pages, poster PDF | Same UTA sky130 300 K/77 K campaign as the data in use; the EDIT 2026 poster states that models will be released on GitHub; no data link. |
| Sky130 RadTol test chip, MPW-4 slot 24 (foss-eda-tools.googlesource.com) | search result README | Total-ionizing-dose test chip; no measurement data link. |
| `msaligane/openfasoc_cryo_caravel` | Git tree | Design files only; no measurements. |
| IHP-Open-PDK branches main and dev, `ihp-sg13cmos5l` | Git tree | No MOS measurement data beyond source 4. |
| IHP 0.13 um SiGe BiCMOS cryogenic PDK paper (publica.fraunhofer.de/handle/publica/512355) | search result | Paper reports 4 K characterization and cryogenic MOSFET models; no data or model release found. |
| ICsprout 55 nm open PDK (icsprout55-pdk-en.readthedocs.io) | documentation site | No measured transistor data published. |
| wafer.space GF180MCU runs | web search | No published transistor test-structure measurements. |
| GitHub repository search | queries: sky130 characterization measurement, sky130 transistor measurement, sky130 iv measurement, sky130 test chip measurements, gf180 measurement, gf180mcu characterization, sg13g2 measurement, mosfet iv measurement data, transistor characterization data keithley, mosfet parameter extraction measured data, bsim4 parameter extraction, cryogenic mosfet, tid mosfet, mdm iccap | `deliciouslytyped/osasic-characterization` is an empty index; `LokeshM01/BSIM4-Parameter-Extraction-...` holds course reports; no other measured multi-device sets. |
| Zenodo API (all types and type=dataset) | sky130 measurement/transistor, skywater 130 characterization, gf180mcu measurement, GF180 transistor, SG13G2 measurement, MOSFET I-V measurement dataset, transistor IV curves dataset, compact model parameter extraction measured, BSIM parameter extraction data, Id-Vg, Id-Vd, MOSFET characterization data, FDSOI cryogenic measurement data, FinFET cryogenic, FDSOI 4 K, MOSFET 4 K, MOSFET 77 K, cryogenic CMOS dataset, cryo-CMOS characterization, transistor 4.2 K, total ionizing dose transistor characterization, 65 nm / 28 nm transistor characterization, bulk CMOS cryogenic, liquid nitrogen transistor characterization | Only source 5 holds CMOS transistor I-V. 22874404 (HKMG defects to cryogenic temperature) and 23466810 (LDMOS hot-carrier stress to deep-cryogenic temperature) are PDFs only; 1299488 is a simulation-to-lookup-table tutorial; 11618610 is a 28 nm 1T2R PCM cell. |
| figshare API | MOSFET I-V, sky130, transistor characteristics, compact model | No CMOS transistor I-V datasets. |
| 4TU.ResearchData API | MOSFET, transistor characterization, sky130, compact model | No transistor I-V datasets beyond the Delft 40 nm set from I3-1. |
| NIST data repository (rmm records) | Sky130, cryogenic CMOS, transistor, METIS | mds2-2997 is the only CMOS transistor I-V record. |
| IEEE DataPort keyword pages | mosfet-modeling, cryogenic | Only ESDED, MESD (sources 7, 8) and the SUSTech set already in I3-1. |

## Ranked shortlist

1. NIST mds2-2997 ETest tile (source 1) paired with
   `google/skywater-pdk-sky130-raw-data` (source 3). 114 sky130 devices have
   Id-Vg and Id-Vd at 4 K and at room temperature under the same module/pad
   IDs. Same process as the 77 K dataset in use, open sky130 BSIM4 cards,
   3.8 MB zipped plus 228 matching room-temperature MDM files (12,293,568
   bytes), MDM format
   (the IHP parser handles MDM). Limits: only 25 of the 114 are 1.8 V core
   devices (5 per type); the room-temperature sweeps have 50 mV steps; die
   identity between the two temperatures is not stated.
2. NIST mds2-2997 MPW-5 tile (source 2). 287 sky130 transistors at 4 K, with
   80 nfet_01v8 and 72 pfet_01v8 W/L combinations from L = 0.15 um to 20 um,
   10 mV Id-Vg steps and five body biases. This is the largest sky130
   geometry set found. Limits: one temperature for the transistor module
   (5 room-temperature reference devices); W/L must be mapped from the layout
   xlsx; two devices are flagged in folder readmes; 39.5 MB.
3. `google/skywater-pdk-sky130-raw-data` alone (source 3). 451 sky130 devices
   at room temperature across 14 device types, Apache-2.0, open cards. It
   gives a room-temperature population for the same process as the 77 K data
   and the NIST 4 K data. Limit: one temperature, 50 mV steps. An
   alternative third choice is the 101 IHP SG13G2 DUTs measured only at
   300 K (source 4), which extend the IHP dataset already in use.
