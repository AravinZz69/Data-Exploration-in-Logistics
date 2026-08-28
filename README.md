# Logistics Data Analysis — Strategic Planning & Data Exploration

A Data-Driven Framework for Inventory, Delivery Performance, and Last-Mile Route Efficiency.

**Prepared By:** KanNa  
**Project Phase:** Week 1 — Strategic Planning & Data Exploration  
**Primary Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, OR-Tools

---

## Project Overview

This project builds a Python-based logistics analytics framework for a regional retail distribution network. It integrates inventory and transportation data to:

- Measure operational KPIs
- Identify causes of inefficiency
- Predict delivery and inventory risks
- Support better resource allocation decisions

---

## Business Scenario

A retail company operates **3 regional distribution centers (DCs)** replenishing stores and e-commerce orders daily using a fleet of vehicles.

**Problems addressed:**
| Problem | Description |
|---|---|
| Inventory | Stockouts at some warehouses while others are overstocked |
| Transportation | Routes misaligned with demand, traffic, and vehicle capacity |
| Service | Late deliveries reducing customer satisfaction |
| Planning | Heavy reliance on historical averages and manual decisions |

---

## KPIs

| KPI | Target |
|---|---|
| On-Time Delivery Rate | Higher |
| Stockout Rate | Lower |
| Inventory Turnover | Higher (within service constraints) |
| Average Delivery Lead Time | Lower |
| Transportation Cost / Delivery | Lower |
| Route Distance per Delivery | Lower |
| Vehicle Utilization | Higher (without service loss) |

---

## Project Structure

```
logistics-data-analysis/
│
├── data/
│   ├── raw/            # Raw downloaded datasets
│   ├── processed/      # Cleaned and integrated tables
│   └── external/       # Reference data (shapefiles, distance matrices)
│
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_eda_and_kpis.ipynb
│   ├── 03_prediction.ipynb
│   ├── 04_clustering.ipynb
│   └── 05_route_optimization.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── features.py
│   ├── kpis.py
│   ├── models.py
│   └── optimization.py
│
├── outputs/
│   ├── figures/        # Charts and visualizations
│   ├── models/         # Saved model files
│   └── reports/        # Final reports and summaries
│
├── requirements.txt
└── README.md
```

---

## Analytical Roadmap

| Phase | Activity | Output |
|---|---|---|
| 1 | Problem Definition | Project charter |
| 2 | Data Acquisition | Raw data + source log |
| 3 | Data Engineering | Cleaned tables |
| 4 | Data Integration | Integrated analytical dataset |
| 5 | Exploratory Data Analysis | EDA notebook |
| 6 | KPI Baseline | KPI baseline |
| 7 | Feature Engineering | Model-ready dataset |
| 8 | Predictive Modeling | Validated prediction model |
| 9 | Segmentation | Customer/Route/SKU segments |
| 10 | Optimization | Optimized routes / allocation scenarios |
| 11 | Evaluation | Performance comparison |
| 12 | Decision Support | Final recommendations |

---

## Data Sources

| Source | Data Type | Use |
|---|---|---|
| [Real-world last-mile delivery dataset (2025)](https://www.sciencedirect.com/science/article/pii/S2352340925004895) | Distance/time matrices; delivery stops | VRP, route cost optimization |
| [Planned vs. driven routes dataset (2026)](https://www.sciencedirect.com/science/article/pii/S2352340926002970) | Planned and actual routes | Route deviation analysis |
| [Kaggle DataCo Global Supply Chain](https://www.kaggle.com/competitions/dataco-global-supply-chain/data) | Orders, shipping, late-delivery info | Late-delivery prediction, EDA |
| [Kaggle Logistics & Supply Chain](https://www.kaggle.com/dsv/9673933) | GPS, fuel, congestion, inventory | Multivariate logistics analysis |

---

## Setup

```bash
pip install -r requirements.txt
```

Then open notebooks in order (01 → 05) in Jupyter Lab or VS Code.

---

## Week 1 Checklist

- [x] Logistics scenario defined
- [x] At least three KPIs defined and justified
- [x] Public data sources identified
- [x] Data science methodologies mapped to business problems
- [x] End-to-end analytical roadmap documented
- [x] Python code illustrations included
- [x] Model evaluation criteria defined
- [x] Risks and assumptions documented
- [x] Expected outcomes and decision impact explained
- [x] References included
