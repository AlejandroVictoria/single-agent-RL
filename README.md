# Single-Agent Tabular Reinforcement Learning for Bus Bunching Mitigation

[![Paper Status](https://img.shields.io/badge/Springer-LNCS-blue.svg)](https://link.springer.com)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation and simulation testbeds for the paper:
> **"Single-Agent Tabular Reinforcement Learning for Bus Bunching Mitigation"**  
> *Alejandro Victoria-Peregrino and Rolando Menchaca-Méndez*  
> Accepted for publication in **Springer Lecture Notes in Computer Science (LNCS)**.  
> **DOI / Paper Link:**

---

## Overview

This repository contains the simulation pipelines, tabular Reinforcement Learning algorithms (Q-Learning, SARSA, Expected SARSA), and evaluation suites used to mitigate bus bunching and optimize headway regularity in microscopic transit network simulations.

---

## Abstract

> Bus bunching has been a continuous problem in urban transit networks, degrading service reliability. This study evaluates the application of Reinforcement Learning (RL) algorithms (Q-Learning, SARSA, and Expected SARSA) to regularize spatial headways and mitigate fleet irregularities on a Trolleybus corridor. Results pointed out that RL agents successfully stabilized the system in a highly stochastic environment. The converged policies reduced safety threshold violations by more than half, elevated service compliance to over 94\%, and decreased the headway coefficient of variation ($C_v$) from 0.2295 to approximately 0.18. The improvements achieved directly benefited passengers by reducing accumulated wait times by 80 seconds. However, a spatial friction analysis revealed that high-friction segments remain structurally fixed, aligning closely with passenger station locations. This suggests that while tabular RL effectively regulates inter-vehicle headways, localized holding interventions operate within the physical constraints of the station layout.

---

## Repository Structure

```text
single-agent-RL/
├── .gitignore
├── README.md
├── requirements.txt
├── agent_training.py
├── parallel_training.py
│
├── graph_analysis/
│   ├── plot_learning.py
│   ├── plot_penalizes.py
│   ├── plot_physics_box.py
│   │
│   └── preprocessing_friction_index/
│       ├── export_data_spatial.py
│       ├── mapeo_estaciones.py
│       │
│       ├── output/
│       │   ├── analisis_friccion_tramos_L8_filtrado.geojson
│       │   └── estaciones_qgis.csv
│       │
│       └── statics/
│           ├── mapeo_estaciones.csv
│           ├── red_zacatenco.geojson
│           └── stops.txt
│
├── monitoring/
│   └── log_class.py
│
├── rl_env/
│   ├── agents.py
│   └── gym_env.py
│
├── sumo_manager/
│   └── sumo_manager.py
│
└── sumo_files/
    ├── adicional/
    │   └── zacatenco_trole_v3.add.xml
    │
    ├── configuracion/
    │   └── zacatenco_v4.sumocfg
    │
    ├── mapas/
    │   ├── zacatenco_v2.poly.xml
    │   └── zacatenco_v4.net.xml
    │
    └── rutas/
        ├── carros_particulares/
        │   ├── config_demandas_v2.xml
        │   ├── trafico_fondo_v4.rou.alt.xml
        │   ├── trafico_fondo_v4.rou.xml
        │   └── trafico_fondo_v4.trips.xml
        │
        ├── peatones/
        │   ├── peatones_intermodales_v4.rou.alt.xml
        │   ├── peatones_intermodales_v4.rou.xml
        │   ├── peatones_intermodales_v4.trips.xml
        │   ├── peatones_intermodales_v5.rou.alt.xml
        │   ├── peatones_intermodales_v5.rou.xml
        │   └── peatones_intermodales_v5.trips.xml
        │
        └── transporte_publico/
            └── zacatenco_trole_v2.rou.xml
```

## 📬 Contact

For questions or inquiries regarding the simulation setup or code implementation, please contact:
- **Alejandro Victoria-Peregrino** - avictoriap2025@cic.ipn.mx - Centro de Investigación en Computación (CIC-IPN)
