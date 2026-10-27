# Literature search, 2026-10-27

Scope: prior work relevant to the project method, for judging overlap.

Project method (reference for the "differs" column):

- Data: SkyWater 130 nm (sky130) 1.8 V LVT transistors measured at 77 K
  (18 devices); NIST sky130 4 K data; IHP SG13G2 (PSP103, 233-398 K).
- Step 1: per-device BSIM4 card fit with NGSpice in the loop: CMA-ES global
  search, then trust-region least squares with finite-difference Jacobians;
  7 or 43 parameters.
- Step 2: one learned correction g shared by all devices:
  `I = I_card * exp(g(x1, x2, is_pmos))`, `x1 = gate drive / VDD`,
  `x2 = |Vds| / VDD` (drain-source symmetric inputs); small MLP ensemble;
  loss = paper RRMS plus a penalty keeping gm/I and gds/I non-negative;
  cards refit with g fixed in the loop; g compiled to Verilog-A
  (OpenVAF / OSDI) and run in NGSpice as an element adding
  `(exp(g) - 1) * I_channel`; a dataset-level gate decides from training
  data whether to apply g.

Procedure: WebSearch (standard and extended), WebFetch on arXiv abstract and
publisher pages, Crossref and OpenAlex metadata queries (title, authors,
venue, DOI, abstract where indexed), and text extraction of two open PDFs
(ASP-DAC 2025 slides of Choi et al.; author manuscript of Gaidhane et al.).

Verification column codes:

- `A` = abstract or full text read (arXiv page, OpenAlex abstract, open PDF).
- `M` = bibliographic metadata confirmed (Crossref/OpenAlex); content from
  search-result snippets only.
- `K` = citation from prior knowledge, not confirmed in this search. Treat as
  uncertain.

## 0. The prior paper and the upstream repository

| item | citation | verification |
|---|---|---|
| Prior paper (77 K sky130 BSIM4 model and measured data) | F. Beall, A. Rimal, O. Seidel, Y. Mei, A. D. McDonald, I. Parmaksiz, V. A. Chirayath, J. Asaadi, D. Braga, J. B. R. Battat, "DC Cryogenic Modeling of Open-Source SkyWater 130 nm MOSFETs at 77 K Using BSIM4," arXiv:2604.21625 (v1 23 Apr 2026; v3 19 Oct 2026), doi:10.48550/arXiv.2604.21625. v1 PDF header: "Prepared for submission to JINST"; no journal reference listed. Models: github.com/UTA-Advanced-Detector-Technologies/Skywater-130nm-77K-Cryogenic-Models | A |
| Upstream ML repository used by this project | `ogzamour/CryoPDK_Skywater130nm_ML` (GitHub; returned HTTP 404 on 2026-10-27, local copy at `data/raw/CryoPDK_Skywater130nm_ML`). The README cites no paper; it states it runs NGSpice with translated models from the UTA repository above. | A (local README) |
| ML pipeline presentation | O. Gzamouranis, "Compact Modeling for CMOS Devices at Cryogenic Temperatures using Machine Learning," talk, New Perspectives 2026, Fermilab, 16 Jul 2026 (indico.fnal.gov/event/73553/contributions/344364). Snippet text: an open-source MLP that predicts best-fit parameters for 77 K I-V curves, trained on NGSpice curves generated around the hand-extracted UTA 77 K card. | M (indico pages returned HTTP 403; content from search snippets) |
| Related press item | M. Bernstein, "Using AI, Fermilab researcher probes how transistors behave in extreme cold," Fermilab News, 26 May 2026. Names "Olivia Seidel"; reports ~120 ms inference; no architecture or accuracy numbers. | A |

Notes on the prior paper (from the v3 HTML):

- Extraction is not ML. Parameters are multiplicative factors (default 1)
  inserted into the sky130 PDK, extracted with Synopsys Mystic 2023.12 in a
  staged procedure: nMOS VTH0, U0, RDSW, NFACTOR from the |Vds| = 0.01 V
  transfer curve, then VSAT; ETA0 only for short devices; DELTA only where
  needed; pMOS VTH0, RDSW, NFACTOR from the low-bias curve, then U0 from the
  |Vds| = 1.85 V curve.
- RRMS per curve = RMS(model - data) / mean(|data|); per transistor = mean
  over curves. Reported average on the order of 20 % RRMS; pMOS
  overpredicted at the lowest drain bias.
- 22 transistors measured at 77 K (8 nMOS, 14 pMOS), 18 distinct 77 K
  models. No ngspice version is stated.
- The 7 tuned parameters in the upstream ML repository (VTH0, U0, NFACTOR,
  VSAT, DELTA, RDSW, ETA0) are the union of the paper's extraction set.
- Whether "O. Seidel" (paper author, press item) and "O. Gzamouranis"
  (talk, GitHub `ogzamour`) are the same person: not confirmed.
- No peer-reviewed or arXiv paper describing the upstream MLP pipeline was
  found. The citable items are the arXiv paper (data, card, metric) and the
  talk (ML pipeline).

Other sky130 cryogenic work:

| citation | what it does | differs from project | ver. |
|---|---|---|---|
| A. Akturk, A. Tripathi, M. Saligane, "Cryogenic Modeling for Open-Source Process Design Kit Technology," IEEE BCICTS 2023, pp. 58-65, doi:10.1109/BCICTS54660.2023.10310944 | Measures sky130 devices at 4 K and fits SPICE-type compact models. Data are the NIST mds2-2997 release (Akturk et al., doi:10.18434/mds2-2997). | 4 K only; conventional fitting; no learned correction. Method details beyond the abstract not read. | A (abstract) |
| M. Montanares, V. H. Arzate Palma, K. G. McCarthy, G. Molina Salgado, "Open-Source 4 K CMOS Calibration: Integrating IceMOS and Sky130 PDK," IFIP/IEEE VLSI-SoC 2025, doi:10.1109/VLSI-SOC64688.2025.11421773 | sky130 at 4 K: SEKV-E extraction of VTH0 and slope factor after a partial Z-score filter, then iterative BSIM4 calibration with IceMOS in a Docker ngspice flow. Median strong-inversion error 1.11 % nMOS, 9.49 % pMOS. | 4 K; staged conventional extraction; no shared correction; error metric is median percent error in strong inversion, not the paper RRMS. | A (abstract) |
| A. Datsuk et al. (IHP), cryogenic PDK for IHP 130 nm SiGe BiCMOS, publica.fraunhofer.de/handle/publica/512355 | Cryogenic PDK modifications (tunneling, HBT temperature effects, poly resistors), validated on ring oscillators and LNAs. | Different PDK; year and MOSFET-model details not confirmed. | M (snippet only) |

## 1. Neural-network compact models (pure ANN, physics-informed, Verilog-A)

| citation | what it does | differs from project | ver. |
|---|---|---|---|
| V. B. Litovski, J. I. Radjenovic, Z. M. Mrcarica, S. L. Milenkovic, "MOS transistor modelling using neural network," Electronics Letters 28:1766-1768, 1992, doi:10.1049/el:19921124 | Early NN I-V model of a MOSFET. | Pure NN; no physics base. | M |
| L. Zhang, M. Chan, "Artificial neural network design for compact modeling of generic transistors," J. Comput. Electron. 16:825-832, 2017, doi:10.1007/s10825-017-0984-9 | ANN structure for generic transistor I-V/Q-V compact models. | Pure NN replaces the compact model; whether it uses symmetric inputs not checked. | M |
| J. Wang, Y.-H. Kim, J. Ryu, C. Jeong, W. Choi, D. Kim, "Artificial Neural Network-Based Compact Modeling Methodology for Advanced Transistors," IEEE TED 68(3):1318-1325, 2021, doi:10.1109/TED.2020.3048918 | Samsung ANN compact-model methodology for DTCO/pathfinding. | Pure NN on TCAD/simulated targets. Related Samsung patents describe input pre-processing and transformed targets (US 11176447); content of the paper's transforms not read. | M |
| C.-T. Tung, C. Hu, "Neural Network-Based BSIM Transistor Model Framework: Currents, Charges, Variability, and Circuit Simulation," IEEE TED 70(4):2157-2160, 2023, doi:10.1109/TED.2023.3244901 | NN models for drain/source/gate currents and charges and their variability; training data generated from BSIM over L, W, Tox. | Replaces BSIM with an NN trained on BSIM output; no measured-data residual. | M |
| C.-T. Tung, S. Salahuddin, C. Hu, "A SPICE-Compatible Neural Network Compact Model for Efficient IC Simulations," SISPAD 2024, doi:10.1109/SISPAD62626.2024.10733248 | IV and QV networks with geometry dependence in Verilog-A; ~40x faster than a conventional compact model. | Pure NN; Verilog-A deployment is the shared element. | M |
| C.-T. Tung, G. Pampori, C. K. Dabhi, S. Salahuddin, C. Hu, "A Novel Neural Network-Based Transistor Compact Model Including Self-Heating," IEEE EDL 45:1512-1515, 2024, doi:10.1109/LED.2024.3408151 | NN compact model with a self-heating node. | Pure NN; room temperature. | M |
| C.-T. Tung, J.-H. Chen, S. Salahuddin, C. Hu, "Physics-Enhanced Neural Network Compact Model for Fast and Flexible Transistor Modeling," IEEE TED 72(12):6487-6493, 2025, doi:10.1109/TED.2025.3627591 | Physical relations built into the NN (zero current at zero bias, variability, post-training fitting flexibility); geometries and temperatures. | NN is the model, physics is structural; no fitted BSIM card underneath. Temperature range not checked. | A (abstract) |
| R. Novkin, H. Amrouch, "Kolmogorov-Arnold Network for Transistor Compact Modeling," Advanced Intelligent Systems 8:2500325, 2025, doi:10.1002/aisy.202500325 | KAN as transistor compact model. | Pure learned model. | M |
| R. Novkin, H. Amrouch, "NN-Based Transistor Modeling for Ultra-Fast SPICE Simulation and Standard Cell Library Characterization," arXiv:2610.31024, 22 Oct 2026 | NN models of Id and charges for a 3 nm nanosheet transistor, used in cell characterization; benchmark vs Verilog-A BSIM-CMG. | Pure NN; simulator integration method not stated in abstract. | A (abstract) |
| G. Guo, H. You, C. Li, Z. Tang, O. Li, "A Physics-Informed Automatic Neural Network Generation Framework for Emerging Device Modeling," Micromachines 14(6):1150, 2023, doi:10.3390/mi14061150 | AutoPINN: monotonic network block with non-negative weights for selected inputs; architecture search; GAA device. | Pure NN with hard monotonicity by weight sign; project uses a soft penalty on gm/I and gds/I of the corrected current. | A (abstract + snippet) |
| H. Yang, Y. Li, S. R. Kola, "A Physical-Based Artificial Neural Networks Compact Modeling Framework for Emerging FETs," IEEE TED 71:223-230, 2024, doi:10.1109/TED.2023.3269410 | ANN compact-model framework for GAA NS / CFET avoiding simulation divergence. | Pure NN on TCAD. | M |
| Y. Zhao et al., "Compact Modeling of Advanced Gate-All-Around Nanosheet FETs Using Artificial Neural Network," Micromachines 15:218, 2024, doi:10.3390/mi15020218 | ANN size/activation study on TCAD NSFET DC/AC data. | Pure NN. | A |
| H. Jeong, J. Choi, ..., S. Kim, "Efficient Neural Network-Based Compact Modeling for Novel Device Structures Using a Multi-Fidelity Model and Active Learning," Electronics 13:4840, 2024, doi:10.3390/electronics13234840 | Multi-fidelity NN compact model, Verilog-A in SPICE. | Pure NN. | M |
| K. V. Aadithya, P. Kuberry, B. Paskaleva, P. Bochev, K. Leeson, A. Mar, T. Mei, E. R. Keiter, "Data-Driven Compact Models for Circuit Design and Analysis," Sandia / OSTI 2020 (also MSML 2020, PMLR 107; venue not re-confirmed) | Data-driven device models inside the Xyce simulator. | Black-box models; not a correction on BSIM. | M |
| A. D. Gaidhane, Z. Yang, Y. Cao, "Graph-based Compact Modeling (GCM) of CMOS Transistors for Efficient Parameter Extraction: A Machine Learning Approach," Solid-State Electronics, 2022 (PII S0038110122003525) | GNN whose nodes are physical quantities (e.g., subthreshold factor from MLP + equation); predicts Ids, gm, gds; Verilog-A; 7 nm FinFET PDK. | Physics graph replaces BSIM equations; trained on PDK simulation, not measured silicon; room temperature. | A (author manuscript) |
| Y. Cho, ..., S. Jung, "Artificial neural networks in semiconductor device compact modeling: A review," Applied Physics Reviews 13:021327, 2026, doi:10.1063/5.0333553 | Review. | Survey; use for coverage check. | M |

## 2. Hybrid physics + NN models, and ML-assisted parameter extraction

### 2a. NN correction or residual on a physics compact model

| citation | what it does | differs from project | ver. |
|---|---|---|---|
| L. Zhang, J. Xu, M. C. E. Yagoub, R. Ding, Q.-J. Zhang, "Neuro-Space Mapping technique for nonlinear device modeling and large signal simulation," IEEE MTT-S IMS 2003, doi:10.1109/MWSYM.2003.1210909 | NN maps terminal voltages/currents between a coarse existing device model and the real device (SiGe HBT, GaAs MESFET). | Correction acts on the coarse model's inputs/outputs via voltage/current mapping, per device; microwave devices; no derivative-sign constraint. | A |
| P. M. Watson, K. C. Gupta, R. L. Mahajan, "Applications of knowledge-based artificial neural network modeling to microwave components," Int. J. RF Microw. CAE 9:254-260, 1999 | "Difference method" (NN learns data minus prior model) and "prior knowledge input" method. | Additive residual on microwave component models; not transistors in SPICE with BSIM. | A |
| S. Li, J. Cheng, B. Han, J. Gao, "Bias-dependent small-signal modeling based on neuro-space mapping for MOSFET," Int. J. RF Microw. CAE, 2011, doi:10.1002/mmce.20502 | Neuro-SM for 130 nm MOSFET small-signal model. | Small-signal RF, per device. | A |
| M.-Y. Kao, H. Kam, C. Hu, "Deep-Learning-Assisted Physics-Driven MOSFET Current-Voltage Model," IEEE EDL 43(6):974-977, 2022, doi:10.1109/LED.2022.3168243 | Part of BSIM-CMG as the physics model plus a 3-layer, 6-neuron NN; TCAD GAAFET; 3-sigma errors 1.3 % (I), 4.1 % (gds), 2.9 % (gm). | NN is embedded inside a reduced BSIM-CMG, not a multiplicative factor on a full fitted card; TCAD data; single technology; exact NN placement not read. | A (abstract) |
| H. Kaneshiro, "Deep-Learning-Assisted MOSFET Current-Voltage Model with Symmetric Fourier Features," TechRxiv preprint, Apr 2024, doi:10.36227/techrxiv.171198080 | Extends the Kao model with symmetric Fourier features designed to satisfy Gummel symmetry; FinFET TCAD; 3-sigma gds error 4.14 %. | Symmetry built into input features (project uses |Vds| inputs); TCAD; not peer-reviewed as found. | M (snippet) |
| J.-Y. Choi, H. Jeong, J.-T. Kong, S. Kim, "Hybrid Compact Modeling Strategy: A Fully-Automated and Accurate Compact Model with Physical Consistency," ASP-DAC 2025, pp. 828-834, doi:10.1145/3658617.3697611 | PMG-NN extracts BSIM-CMG parameters; DGC-NN is a global correction trained on target `Y_TCAD / Y_BSIM` over 100 process-varied devices; inputs Lg, W_NS, T_NS, N_SD, N_body, Vgs, Vds; loss `mean((eps(X) * Y_BSIM - Y_TCAD)^2)`; correction applied at the final stage of BSIM-CMG in Verilog-A; Gummel symmetry test and SPICE validation. | Closest structural match (see Section 6a). Differences: TCAD 3 nm NSFET data, not measured; correction conditioned on process/geometry inputs; raw Vgs/Vds inputs; MSE loss on linear current, not a relative RRMS loss; no gm/gds sign penalty reported; no card refit with the correction in the loop reported; no apply/skip gate; room temperature. | A (abstract + slide text) |
| S. Eom, S. Lee, H. Yun, K. Cho, S.-M. Kim, R.-H. Baek, "Machine Learning-Driven Extraction of Hybrid Compact Models Integrating Neural Networks and BSIM-CMG for Multidevice Applications," Advanced Intelligent Systems, 2024/2025, doi:10.1002/aisy.202400571 | BSIM-CMG plus an error-correction NN for Id and gate capacitance; extraction networks set both models' parameters for multiple devices at once; TCAD 3 nm NSFET; MAPE 1.8 % Id, 2.8 % gm, 8.5 % gds. | Multi-device hybrid; TCAD; correction form (additive / multiplicative) not read; SPICE deployment not confirmed. | A (abstract) |
| J. Deng, L. Ma, J. Xu, A. Petr, "ANN-Assisted SPICE Model Solutions," Int. J. Numer. Model. 2025, doi:10.1002/jnm.70065 | ANN enhancement on top of a physical SPICE model (hybrid), plus ANN-derived inductor scaling equations. | Form of the hybrid not read. | A (abstract) |
| Q. Yang, J. Guo, "A machine-learning virtual source model for nanoscale transistors," Scientific Reports, 2026, doi:10.1038/s41598-026-47672-7 | End-to-end trainable NN + virtual-source model; measured 2D FETs and Si FinFETs; snippet reports Gummel symmetry checks to 4th derivative via mirroring for negative Vds. | NN integrated into VS equations (not BSIM); per-technology training. | A (abstract) |
| Balamanikandan A. et al., "Physics-Informed Neural Network-Assisted Compact Modeling of UTB-SOI and Nanowire MOSFETs for Ultra-Low Power Edge-AI Applications," IJEEE 22(2):119-135, 2026 | PINN correction on a baseline compact model from TCAD; 273-373 K and radiation; log-current targets; ONNX export. | TCAD; extracted BSIM parameters for SPICE rather than the NN in SPICE. | A (abstract) |
| S. Huang, L. Wang, "MOSFET Physics-Based Compact Model Mass-Produced: An Artificial Neural Network Approach," Micromachines 14:386, 2023, doi:10.3390/mi14020386 | ANN learns surface potential within BSIM-CMG; argues that a residual on (data - classical model) carries little physical content; Verilog-A. | Explicitly argues against the residual approach the project uses. | M (snippet) |

### 2b. ML-assisted or global-optimizer parameter extraction

| citation | what it does | differs from project | ver. |
|---|---|---|---|
| M.-Y. Kao, F. Chavez, S. Khandelwal, C. Hu, "Deep Learning-Based BSIM-CMG Parameter Extraction for 10-nm FinFET," IEEE TED 69:4765-4768, 2022, doi:10.1109/TED.2022.3181536 | Inverse NN: curves -> BSIM-CMG parameters. | Amortized inverse map; project fits each card directly with NGSpice in the loop. | M |
| F. Chavez, C.-T. Tung, M.-Y. Kao, C. Hu, J.-H. Chen, S. Khandelwal, "Deep learning-based I-V Global Parameter Extraction for BSIM-CMG," Solid-State Electronics, 2023, doi:10.1016/j.sse.2023.108766 | Global (multi-L) inverse NN; 300k Monte Carlo training sets; TCAD 10 nm FinFET. | As above. | A |
| J.-H. Chen, F. Chavez, C.-T. Tung, S. Khandelwal, C. Hu, "A single neural network global I-V and C-V parameter extractor for BSIM-CMG compact model," Solid-State Electronics 216:108898, 2024 | Single NN for I-V and C-V global extraction. | As above. | M |
| R. Ashai et al., "Deep Learning-Based Fast BSIM-CMG Parameter Extraction for General Input Dataset," IEEE TED 70:3437-3441, 2023, doi:10.1109/TED.2023.3278615 | Cascade of inverse and forward NNs for BSIM-CMG extraction. | As above. | M |
| B. Guo et al., "GatedNN: An accurate deep learning-based parameter extraction for BSIM-CMG," Solid-State Electronics 224:109044, 2025 | Gated NN inverse extractor. | As above. | M |
| S. Lee, S. Eom, ..., R.-H. Baek, "Multi-Task Learning for Real-Time BSIM-CMG Parameter Extraction of NSFETs With Multiple Structural Variations," IEEE Access, 2024, doi:10.1109/ACCESS.2024.3512612 | Multi-task inverse extractor. | As above. | M |
| Kang et al., "Machine Learning-Based Standard Compact Model Binning Parameter Extraction Methodology ...," Advanced Intelligent Systems 8:e202500511, 2025 | ML binning-parameter extraction. | Binning; sky130 cards are binned, project tunes multipliers on given bins. | M |
| R. Lyu, Z. Bi, K. Zhu, Y. Lu, X. Zeng, "S3HDBO: Sensitivity-Guided Search Region Scaling High-Dimensional Bayesian Optimization for BSIM-CMG Parameter Extraction," IEEE TED, 2026, doi:10.1109/TED.2026.3665864 | High-dimensional BO for BSIM-CMG extraction with sensitivity pruning. | BO instead of CMA-ES + trust-region LS; no learned correction. | A (abstract) |
| E. Papageorgiou, G. Alia, A. Buzo, G. Pelz, T. Noulis, "MOSFET Model Parameter Extraction Using Reinforcement Learning," PACET 2024, doi:10.1109/PACET60398.2024.10497080 | RL agent with continuous actions for BSIM parameter extraction across reference datasets. | RL; no correction model. | A (abstract) |
| Son et al., "Bayesian Optimization-Reinforcement Learning Hybrid Framework for a-IGZO Thin-Film Transistor Compact Model Parameter Extraction ...," Advanced Intelligent Systems, 2026, doi:10.1002/aisy.70525 | BO + RL extraction for TFTs. | TFT; no correction model. | M |
| C. Bittner, D. Heaberlin, J. Hoffman, J. Watts, "Extraction of Compact Model Parameters for ULSI MOSFETs Using a Genetic Algorithm," TechConnect Briefs, 1999 | GA extraction. | GA vs CMA-ES; same direct-fit family. | M |
| K. Mehta, H.-Y. Wong, "Prediction of FinFET Current-Voltage and Capacitance-Voltage Curves Using Machine Learning With Autoencoder," IEEE EDL 42:136-139, 2021, doi:10.1109/LED.2020.3045064 | Autoencoder-based forward prediction of I-V/C-V from device parameters. | Forward surrogate; not extraction against measured cryogenic data. | M |
| A. Jarndal, S. Husain, M. Hashmi, F. M. Ghannouchi, "Large-Signal Modeling of GaN HEMTs Using Hybrid GA-ANN, PSO-SVR, and GPR-Based Approaches," IEEE JEDS, 2020, doi:10.1109/JEDS.2020.3035628 | GA-initialized ANN, PSO-SVR, GPR current models embedded in an ADS large-signal circuit model. | Black-box current model per device; GaN HEMT. | A |

No paper using CMA-ES followed by finite-difference trust-region least
squares on BSIM4 with the simulator in the loop was found; CMA-ES and
least squares are standard tools and this combination is not presented as a
contribution.

## 3. Cryogenic CMOS compact modeling

### 3a. Physics and compact models

| citation | what it does | differs from project | ver. |
|---|---|---|---|
| A. Beckers, F. Jazaeri, C. Enz, "Characterization and Modeling of 28-nm Bulk CMOS Technology Down to 4.2 K," IEEE JEDS 6:1007-1018, 2018, doi:10.1109/JEDS.2018.2817458 | 28 nm bulk characterization and modeling (EKV-based) to 4.2 K. | Physics model of effects; no learned correction. | M |
| A. Beckers, F. Jazaeri, C. Enz, "Cryogenic MOS Transistor Model," IEEE TED 65:3617-3625, 2018, doi:10.1109/TED.2018.2854701 | Cryogenic MOSFET model including freeze-out and incomplete ionization. | As above. | M |
| A. Beckers, F. Jazaeri, C. Enz, "Theoretical Limit of Low Temperature Subthreshold Swing in Field-Effect Transistors," IEEE EDL 41:276-279, 2020, doi:10.1109/LED.2019.2963379 | Subthreshold-swing saturation tied to band-tail decay. | Explains one effect a learned g may absorb; analytical. | A |
| A. Beckers, F. Jazaeri, A. Grill, N. Subramanian (as indexed by OpenAlex), B. Parvais, C. Enz, "Physical Model of Low-Temperature to Cryogenic Threshold Voltage in MOSFETs," IEEE JEDS 8, 2020, doi:10.1109/JEDS.2020.2989629 | Vth(T) to 4.2 K with interface traps near band edge; pMOS-specific gate-stack effect. | Analytical Vth model. | A |
| H. Bohuslavskyi et al., "Cryogenic Subthreshold Swing Saturation in FD-SOI MOSFETs Described With Band Broadening," IEEE EDL 40:784-787, 2019, doi:10.1109/LED.2019.2903111 | Band broadening explanation of SS saturation. | Analytical. | M |
| R. M. Incandela, L. Song, H. Homulle, E. Charbon, A. Vladimirescu, F. Sebastiano, "Characterization and Compact Modeling of Nanometer CMOS Transistors at Deep-Cryogenic Temperatures," IEEE JEDS 6:996-1006, 2018, doi:10.1109/JEDS.2018.2821763 | 40/160 nm characterization at 4 K; adapted PSP compact model. | Hand-adapted compact model; no learned correction. | M |
| H. Homulle, L. Song, E. Charbon, F. Sebastiano, "The Cryogenic Temperature Behavior of Bipolar, MOS, and DTMOS Transistors in Standard CMOS," IEEE JEDS 6:263-270, 2018, doi:10.1109/JEDS.2018.2798281 | Device behavior at cryogenic temperature. | Characterization. | M |
| E. Charbon et al., "Cryo-CMOS for quantum computing," IEDM 2016, 13.5.1-13.5.4, doi:10.1109/IEDM.2016.7838410 | Cryo-CMOS overview incl. device modeling needs. | Overview. | M |
| G. Pahwa, P. Kushwaha, A. Dasgupta, S. Salahuddin, C. Hu, "Compact Modeling of Temperature Effects in FDSOI and FinFET Devices Down to Cryogenic Temperatures," IEEE TED 68:4223-4230, 2021, doi:10.1109/TED.2021.3097971 | Temperature-effect extensions in BSIM-IMG / BSIM-CMG down to cryogenic T. | Equation extensions of BSIM; the project keeps stock BSIM4 and adds a learned factor. | M |
| A. Akturk et al., "Compact and Distributed Modeling of Cryogenic Bulk MOSFET Operation," IEEE TED 57:1334-1342, 2010, doi:10.1109/TED.2010.2046458 | Cryogenic bulk MOSFET compact and distributed modeling. | Physics-based. | M |
| A. Akturk, K. Eng, J. Hamlet, S. Potbhare, E. Longoria, R. Young, M. Peckerar, T. Gurrieri, M. S. Carroll, N. Goldsman, "Compact Modeling of 0.35 micron SOI CMOS Technology Node for 4 K DC Operation using Verilog-A," arXiv:1001.3353, 2010 | Modifies BSIM equations in Verilog-A for 4 K SOI, including multiplicative bias-dependent correction functions for kinks; one library across sizes. | Hand-crafted analytic correction functions, not learned; SOI 4 K. | A |
| W. Triantopoulos et al., "Self-Heating Effect in FDSOI Transistors Down to Cryogenic Operation at 4.2 K," IEEE TED 66:3498-3505, 2019, doi:10.1109/TED.2019.2919924 | Self-heating at 4.2 K. | Effect characterization; the project uses isothermal DC fits. | M |
| C. Luo, Z. Li, T.-T. Lu, J. Xu, G.-P. Guo, "MOSFET characterization and modeling at cryogenic temperatures," Cryogenics 98:12-17, 2019, doi:10.1016/j.cryogenics.2018.12.009 | SMIC 0.18 um, 300-4.2 K, BSIM3v3-based model; kink handled by a substrate resistor subcircuit. | Hand model edits. | M |
| T. Lu, Y. Zhang, Y. Zhang, J. Xu, G. Guo, C. Luo, "Cryogenic Modeling of MOSFET Device Based on BSIM and EKV Models," arXiv:2104.08467, 2021 | Kink corrections: modified evolution strategy for impact ionization / substrate effects, and a dual BSIM3v3 + EKV2.6 model. | Evolution-strategy parameter fit (related to CMA-ES use); no learned shared correction. | A |
| Y. Zhang, T. Lu, W. Wang, Y. Zhang, J. Xu, C. Luo, G. Guo, "Characterization and Modeling of Native MOSFETs Down to 4.2 K," IEEE TED 2021, doi:10.1109/TED.2021.3099775 (arXiv:2104.03094) | Modified EKV mobility equations, parameters extracted with an ML optimizer. | EKV; native devices; ML used as optimizer only. | A |
| H. Zhao, X. Liu, "Modeling of a standard 0.35 um CMOS technology operating from 77K to 300K," Cryogenics 59:49-59, 2014, doi:10.1016/j.cryogenics.2013.10.003 | Temperature-dependent functions for BSIM parameters 77-300 K. | Hand-built T-functions; same temperature (77 K) as the project. | M |
| B. Kabaoglu, M. Berke Yelten, "A cryogenic modeling methodology of MOSFET I-V characteristics in BSIM3," SMACD 2017, doi:10.1109/SMACD.2017.7981578 | BSIM3 cryogenic I-V methodology. | Conventional. | M |
| A. Kamgar, "Subthreshold behavior of silicon MOSFETs at 4.2 K," Solid-State Electronics 25:537-539, 1982, doi:10.1016/0038-1101(82)90052-1 | Early SS-saturation observation (cited by the prior paper). | Physics. | M (via prior paper) |
| Z. Tang et al., "Generic Cryo-CMOS Device Modeling and EDA-Compatible Platform for Reliable Cryogenic IC Design," arXiv:2211.05309, 2022/2024 | Cryo-CMOS database, compact model in a PDK, circuit demos. | Model type not stated in abstract. | A |
| T. Dutta, F. Adamu-Lema, D. Bensouiah, A. Asenov, "A Methodology for Process Design Kit Re-Centering Using TCAD and Experimental Data for Cryogenic Temperatures," Solid-State Electronics 232:109306, 2026 (arXiv:2502.02685) | TCAD calibrated to room-temperature and cryogenic data, then compact models for all corners from TCAD targets. | TCAD-mediated PDK re-centering; no learned correction. | A |
| D. Dutta, K. Ture, F. Olivieri, A. Gomez-Saiz, G. M. Noah, "Deep-Cryogenic Modeling of 22-nm FDSOI MOSFETs based on BSIM-IMG," arXiv:2511.22297, 2025 | BSIM-IMG deep-cryogenic model with mobility edits for intersubband scattering. | Equation edits. | A |

### 3b. ML in cryogenic compact modeling

| citation | what it does | differs from project | ver. |
|---|---|---|---|
| Q. Wang, M. Ye, Y. Li, X. Zheng, J. He, J. Du, Y. Zhao, "MOSFET modeling of 0.18 um CMOS technology at 4.2K using BP neural network," Microelectronics Journal 132:105678, 2023, doi:10.1016/j.mejo.2022.105678 | SMIC 0.18 um at 4.2 K: revised BSIM parameters, then a BP-NN "optimization model" calibrates remaining cryogenic effects; the NN is implemented in Verilog-A as a voltage-controlled current source; covers several aspect ratios. | Closest cryogenic match for the SPICE structure (see 6a). Differences as far as known: additive VCCS (project: current element `(exp(g)-1)*I_channel`, i.e., multiplicative); BP-NN inputs, sharing across devices, loss, derivative constraints, and card refit not confirmed (full text not accessible); 4.2 K, single process. | M (abstract via snippets) |
| T. Inaba, Y. Chiashi, H. Oka, M. Ogura, H. Asai, S. Iizuka, K. Kato, S. Shitakata, H. Fuketa, T. Mori, "Neural-network-based prediction of cryogenic BSIM4 model parameters from small datasets," Jpn. J. Appl. Phys. 63(12):12SP19, 2024, doi:10.35848/1347-4065/ad9c83 | 4 K bulk MOSFETs: transfer-learned NN predicts Id for other L; cGAN gives 9 BSIM4 parameter candidates; Levenberg-Marquardt polish; 6.6 % error vs 7.6 % manual. | Amortized initial guess + LM polish for BSIM4; no correction term. Same two-stage "global guess then LS polish" pattern as Step 1. | A |
| T. Inaba, H. Fuketa, H. Oka, Y. Chiashi, H. Asai, S. Iizuka, K. Kato, T. Mori, "BSIM4 modeling of drain voltage, gate length, and temperature-dependent Id-Vg characteristics of bulk MOSFETs at cryogenic temperature using a conditional diffusion neural network," Jpn. J. Appl. Phys. 65(4):044001, 2026, doi:10.35848/1347-4065/ae430a | Conditional diffusion model predicts 15 BSIM4 parameters for 90-250 nm, 4-100 K; LM polish; 6.3 % average error; a snippet reports that SS could not be captured by BSIM4 alone and cryogenic correction terms were added (not confirmed in abstract). | Parameter prediction; correction terms (if any) are hand-designed, not a learned shared factor. | A (abstract) |

## 4. ML methods for simulator misspecification

| citation | what it does | relation to project | ver. |
|---|---|---|---|
| M. C. Kennedy, A. O'Hagan, "Bayesian Calibration of Computer Models," JRSS-B 63:425-464, 2001, doi:10.1111/1467-9868.00294 | Calibration with additive GP discrepancy `y = eta(x, theta) + delta(x) + e`. | g is a discrepancy term, multiplicative in current (additive in log current), point-estimated, shared across devices, and deployed inside the simulator. KOH discrepancy is per-system and probabilistic. | M |
| D. Higdon, J. Gattiker, B. Williams, M. Rightley, "Computer Model Calibration Using High-Dimensional Output," JASA 103:570-583, 2008, doi:10.1198/016214507000000888 | KOH with basis-represented high-dimensional output and discrepancy. | As above. | M |
| J. Brynjarsdottir, A. O'Hagan, "Learning about physical parameters: the importance of model discrepancy," Inverse Problems 30:114007, 2014, doi:10.1088/0266-5611/30/11/114007 | Shows omitting discrepancy biases physical parameters; priors on delta matter. | Relevant to refitting cards with g in the loop (card parameters shift when g absorbs discrepancy). | M |
| R. Tuo, C. F. J. Wu, "Efficient calibration for imperfect computer models," Ann. Statist. 43:2331-2352, 2015, doi:10.1214/15-AOS1314 | L2 calibration; identifiability of theta vs discrepancy. | Identifiability between card parameters and g. | M |
| M. Plumlee, "Bayesian Calibration of Inexact Computer Models," JASA 112:1274-1285, 2017, doi:10.1080/01621459.2016.1211016 | Discrepancy prior orthogonal to the model gradient in theta, for identifiability. | An orthogonality constraint of this kind is not used in the project. | M |
| K. N. Maupin, L. P. Swiler, "Model discrepancy calibration across experimental settings," Reliability Eng. & System Safety 200:106818, 2020, doi:10.1016/j.ress.2020.106818 | Embeds discrepancy inside the simulator so it transfers across experimental settings. | Closest statistical analogue to a shared, embedded discrepancy; GP-based, not deployed in SPICE. | M (snippet) |
| K. Cranmer, J. Brehmer, G. Louppe, "The frontier of simulation-based inference," PNAS 117:30055-30062, 2020, doi:10.1073/pnas.1912789117 | SBI review. | Background for amortized parameter inference (the upstream MLP). | M |
| P. Cannon, D. Ward, S. M. Schmon, "Investigating the Impact of Model Misspecification in Neural Simulation-based Inference," arXiv:2209.01845, 2022 | Neural SBI fails under misspecification; no tested mitigation prevents all failures. | Motivates not trusting amortized inverse maps when BSIM4 cannot reproduce the data. | A |
| D. Ward, P. Cannon, M. Beaumont, M. Fasiolo, S. M. Schmon, "Robust Neural Posterior Estimation and Statistical Model Criticism," NeurIPS 35, 2022 | RNPE: explicit error model on summary statistics. | Error model on summaries, not a deployable correction. | M |
| D. Huang, A. Bharti, A. Souza, L. Acerbi, S. Kaski, "Learning Robust Statistics for Simulation-based Inference under Model Misspecification," NeurIPS 36, 2023 | Regularizes summary statistics to be robust to misspecification. | Same distinction. | M |
| M. Schmitt, P.-C. Burkner, U. Kothe, S. T. Radev, "Detecting Model Misspecification in Amortized Bayesian Inference with Neural Networks," GCPR 2023 (LNCS), doi:10.1007/978-3-031-54605-1_35 | Misspecification detection via summary-space MMD. | Related to the dataset-level gate (decide whether a correction is warranted), but detection differs. | M |
| A. Wehenkel, J. L. Gamella, O. Sener, J. Behrmann, G. Sapiro, J.-H. Jacobsen, M. Cuturi, "Addressing Misspecification in Simulation-based Inference through Data-driven Calibration," ICML 2025 (PMLR 267), arXiv:2405.08719 | RoPE: OT between real and simulated representations using a small labeled calibration set. | Requires real observations with known parameters; project has none. | A |
| R. P. Kelly, D. J. Warne, D. T. Frazier, D. J. Nott, M. U. Gutmann, C. Drovandi, "Simulation-based Bayesian inference under model misspecification," arXiv:2503.12315, 2025 | Review of misspecification-robust SBI. | Survey. | A |
| D. C. Psichogios, L. H. Ungar, "A hybrid neural network-first principles approach to process modeling," AIChE J. 38:1499-1511, 1992, doi:10.1002/aic.690381003 | Early grey-box hybrid (NN inside first-principles model). | Grey-box precedent. | M |
| C. Rackauckas et al., "Universal Differential Equations for Scientific Machine Learning," arXiv:2001.04385, 2020 | NN terms inside mechanistic ODEs. | Grey-box precedent; dynamical systems. | M |
| Y. Yin, V. Le Guen, J. Dona, E. de Bezenac, I. Ayed, N. Thome, P. Gallinari, "Augmenting physical models with deep networks for complex dynamics forecasting" (APHYNITY), ICLR 2021; J. Stat. Mech. 2021:124012, doi:10.1088/1742-5468/ac3ae5 | Physics + NN residual with minimum-norm residual so physics explains as much as possible. | Same identifiability idea as refitting the card first and keeping g small; project uses a gate rather than a norm penalty. | M |
| N. Takeishi, A. Kalousis, "Physics-Integrated Variational Autoencoders for Robust and Interpretable Generative Modeling," NeurIPS 2021, arXiv:2102.13156 | Regularizes trainable part so incomplete physics keeps its meaning. | Same balance question. | A |
| Y. Yin, I. Ayed, E. de Bezenac, N. Baskiotis, P. Gallinari, "LEADS: Learning Dynamical Systems that Generalize Across Environments," NeurIPS 2021, arXiv:2106.04546 | Shared model + environment-specific terms across environments. | Closest ML analogue of "shared g + per-device card": shared component across instances, per-instance component. LEADS learns both parts as NNs; project's per-instance part is a physical BSIM4 card. | A |
| M. Kirchmeyer, Y. Yin, J. Dona, N. Baskiotis, A. Rakotomamonjy, P. Gallinari, "Generalizing to New Physical Systems via Context-Informed Dynamics Model" (CoDA), ICML 2022, arXiv:2202.01889 | Hypernetwork-conditioned dynamics with per-environment context vectors. | Per-instance adaptation via context, not via a physical card. | A |
| A. Zeng et al., "TossingBot: Learning to Throw Arbitrary Objects with Residual Physics," RSS 2019, doi:10.15607/RSS.2019.XV.004 | Learned residual on a physics controller. | Residual-physics precedent (robotics). | M |
| A. Ajay et al., "Augmenting Physical Simulators with Stochastic Neural Networks: Case Study of Planar Pushing and Bouncing," IROS 2018, doi:10.1109/IROS.2018.8593995 | NN residual on an analytical simulator. | Residual-physics precedent. | M |
| J. Willard, X. Jia, S. Xu, M. Steinbach, V. Kumar, "Integrating Scientific Knowledge with Machine Learning for Engineering and Environmental Systems," ACM Computing Surveys 55, 2022, doi:10.1145/3514228 | Survey of physics-guided ML incl. residual modeling. | Survey. | M |

## 5. Derivative / monotonicity / symmetry constraints and NN deployment in SPICE

| citation | what it does | relation to project | ver. |
|---|---|---|---|
| C. C. McAndrew, "Validation of MOSFET Model Source-Drain Symmetry," IEEE TED 53:2202-2206, 2006, doi:10.1109/TED.2006.881005 | Gummel symmetry test and related symmetry checks. | The |Vds| input makes g even in Vds; a GST of the deployed model has not been reported by the project (check). | M |
| K. Joardar, K. K. Gullapalli, C. C. McAndrew, M. E. Burnham, A. Wild, "An improved MOSFET model for circuit simulation," IEEE TED 45(1), 1998 | Commonly cited origin of the Gummel symmetry test. | Not confirmed in this search. | K |
| C. C. McAndrew et al., "Best Practices for Compact Modeling in Verilog-A," IEEE JEDS 3:383-396, 2015, doi:10.1109/JEDS.2015.2455342 | Verilog-A coding practices (derivatives, continuity). | Applies to the compiled g element. | M |
| P. Kuthe, M. Muller, M. Schroter, "VerilogAE: An Open Source Verilog-A Compiler for Compact Model Parameter Extraction," IEEE JEDS 8:1416-1423, 2020, doi:10.1109/JEDS.2020.3023165 | Predecessor of OpenVAF. | Toolchain. | M |
| ngspice OSDI / OpenVAF documentation, ngspice.sourceforge.io/osdi.html | OSDI loading of OpenVAF-compiled Verilog-A in ngspice >= 39. | Toolchain. | A |
| H. Kam (Stanford CS230 project report, winter 2021), "Deep-Learning Assisted Compact Modeling of Nanoscale Transistor" | Loss terms on Ids, gm, gds; correction symmetric by design; passes Gummel symmetry tests to higher derivatives. | Derivative-matching loss (not sign constraints). Not peer-reviewed. | M (snippet) |
| Choi et al. ASP-DAC 2025 (Section 2a) | Hybrid model validated by GST in Verilog-A. | See 6a. | A |
| Guo et al. Micromachines 2023 (Section 1) | Hard monotonicity by non-negative weights. | Hard vs soft constraint. | A |
| J. Sill, "Monotonic Networks," NeurIPS 1997 | Monotone min-max networks. | Hard monotonicity. | K |
| A. Wehenkel, G. Louppe, "Unconstrained Monotonic Neural Networks," NeurIPS 2019, arXiv:1908.05164 | Monotone by integrating a positive-output NN. | Alternative to the project's soft penalty. | A |
| X. Liu, X. Han, N. Zhang, Q. Liu, "Certified Monotonic Neural Networks," NeurIPS 2020, arXiv:2011.10219 | Monotonicity regularization plus MILP certification. | Could certify the sign of dI/dV of the corrected model; project uses a penalty on sampled points only. | A |
| D. Runje, S. M. Shankaranarayana, "Constrained Monotonic Neural Networks," ICML 2023 | Monotone dense layers. | Hard monotonicity. | M |
| W. M. Czarnecki, S. Osindero, M. Jaderberg, G. Swirszcz, R. Pascanu, "Sobolev Training for Neural Networks," NeurIPS 2017, arXiv:1706.04859 | Fits target derivatives as well as values. | Project constrains derivative sign, not derivative values. | A |
| Kao et al. EDL 2022; Eom et al. 2024; Gaidhane et al. 2022 (above) | Report gm/gds accuracy as validation targets. | Accuracy of derivatives, not sign constraints. | A/M |

No paper found that trains a device-model correction with an explicit
non-negativity penalty on gm/I and gds/I of the corrected current. The
Micromachines 2023 AutoPINN uses hard weight-sign monotonicity on a pure NN;
the Stanford report uses gm/gds value losses. This absence is a search
result, not a proof.

## 6. Closest prior work

### 6a. Shared multiplicative correction with derivative constraints deployed in SPICE

1. Choi, Jeong, Kong, Kim, ASP-DAC 2025 (doi:10.1145/3658617.3697611).
   - Same: a physics compact model (BSIM-CMG) with parameters extracted
     first; one correction network trained across many devices on the ratio
     `Y_target / Y_BSIM` and applied multiplicatively,
     `Y = eps(X) * Y_BSIM`; correction placed at the output of the compact
     model in Verilog-A and simulated in SPICE; Gummel symmetry test.
   - Differs: TCAD targets (3 nm NSFET, 100 process-varied devices), not
     measured silicon; correction conditioned on geometry and doping as
     inputs; raw Vgs, Vds inputs (symmetry handled by testing, not by input
     construction); squared-error loss, not a relative RRMS loss; no
     gm/gds sign penalty reported; correction output is `eps` directly, not
     `exp(g)` (form of the NN output layer not read); no refit of cards with
     the correction in the loop reported; no dataset-level apply/skip gate;
     room temperature; also corrects C-V.
2. Q. Wang et al., Microelectronics Journal 132:105678, 2023
   (doi:10.1016/j.mejo.2022.105678).
   - Same: cryogenic (4.2 K) measured MOSFETs; BSIM parameters revised first;
     an NN term calibrates the remaining cryogenic error; NN in Verilog-A as
     a current-source element in SPICE; several aspect ratios.
   - Differs (as far as known): additive VCCS; 0.18 um SMIC; inputs, sharing
     across devices, loss, derivative handling, and card refit not confirmed
     because the full text was not accessible. Treat the overlap level as
     uncertain.
3. Eom et al., Advanced Intelligent Systems, doi:10.1002/aisy.202400571.
   - Same: BSIM-CMG plus error-correction NN fit for multiple devices.
   - Differs: TCAD; correction form and SPICE deployment not confirmed.
4. Kao, Kam, Hu, IEEE EDL 2022 (doi:10.1109/LED.2022.3168243) and Kaneshiro
   2024 (TechRxiv).
   - Same: small NN combined with BSIM-CMG physics; gm/gds accuracy
     reported; symmetric input features (Kaneshiro).
   - Differs: NN embedded within a partial BSIM-CMG for one technology; TCAD.
5. Akturk et al., arXiv:1001.3353, 2010.
   - Same: multiplicative bias-dependent correction factors on BSIM in
     Verilog-A at cryogenic temperature, one library across sizes.
   - Differs: hand-designed analytic functions, not learned; 4 K SOI.

### 6b. Cryogenic sky130 fit

1. Beall et al., arXiv:2604.21625 (the prior paper).
   - Same: sky130 LVT devices at 77 K; BSIM4 multipliers on the sky130 PDK
     cards; same measured data and device list; same RRMS definition.
   - Differs: Mystic staged extraction of 4-6 parameters per device; no
     learned correction; no NGSpice-in-loop CMA-ES/least-squares.
2. Gzamouranis, New Perspectives 2026 talk; upstream repository
   `ogzamour/CryoPDK_Skywater130nm_ML`.
   - Same: sky130 77 K, NGSpice-41, 7 BSIM4 parameters, MLP.
   - Differs: MLP maps I-V to parameters (amortized inverse); trained on
     curves simulated around the hand card; no correction term. No paper
     found.
3. Akturk, Tripathi, Saligane, BCICTS 2023; Montanares et al., VLSI-SoC
   2025.
   - Same: sky130, cryogenic (4 K), BSIM4-based SPICE models; Montanares
     uses ngspice.
   - Differs: 4 K not 77 K; conventional extraction; no learned correction.
4. Inaba et al., JJAP 2024 and 2026.
   - Same: cryogenic BSIM4 parameter extraction with an ML global stage and
     a least-squares (LM) polish.
   - Differs: non-sky130 bulk devices at 4-100 K; inverse/generative NN for
     initial guesses instead of CMA-ES with NGSpice in the loop; no shared
     correction.

## 7. Is the exact method published?

Answer: no paper found that contains the full combination. Partial overlap:

- Shared multiplicative correction on a fitted compact model, trained on
  target/model ratios across devices, in Verilog-A: yes in Choi et al.
  ASP-DAC 2025 (TCAD, room temperature, no derivative-sign penalty reported).
- NN correction on BSIM for cryogenic measured devices in Verilog-A: yes in
  Q. Wang et al. 2023 (4.2 K, additive VCCS; other details unknown).
- Not found in any paper: (i) the `exp(g)` form with drain-source symmetric
  normalized inputs `(gate drive / VDD, |Vds| / VDD, is_pmos)` shared across
  nMOS and pMOS; (ii) training on the paper RRMS loss with a non-negativity
  penalty on gm/I and gds/I of the corrected current; (iii) refitting each
  card with g fixed inside the NGSpice loop; (iv) a dataset-level gate
  deciding from training data whether to apply g; (v) any learned correction
  on sky130 at 77 K or on the NIST 4 K data; (vi) OpenVAF/OSDI deployment of
  a learned device correction in ngspice.

Limits: full texts of Wang 2023, Eom 2024, Kao 2022 and Wang 2021 were not
read (paywalled); IEEE Xplore and ScienceDirect pages returned 403. Overlap
statements for those rely on abstracts and search snippets.

## 8. Own prior artifacts

- `github.com/aaronw5/Cryo-Transistor-ML` (this project's public repository)
  is indexed by web search and describes the 77 K sky130 emulator / direct
  MLP / FD pipeline. It is not independent prior work.
