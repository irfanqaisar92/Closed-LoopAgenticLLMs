# Closed-Loop Agentic LLMs for Occupant-Centric HVAC Supervisory Control

This repository contains the **Python code and simulation files used in the paper**:

**"Closed-Loop Agentic LLMs for Occupant-Centric HVAC Supervisory Control in Smart Buildings"**  
Irfan Qaisar, Kailai Sun, Qianchuan Zhao

---

## 📖 Overview

This project develops a **closed-loop evaluator-guided LLM framework for occupant-centric HVAC supervisory control** using EnergyPlus.

At each **5-minute control interval**, the framework uses the current multi-zone state, including occupancy, temperature, relative humidity, mean radiant temperature, PMV, and previous thermostat setpoints.

The control workflow is:

- **LLM controller:** generates heating and cooling setpoints.
- **LLM evaluator:** reviews the proposed actions for comfort, humidity, energy proportionality, and action consistency.
- **Controller refinement:** rejected actions may be revised once using evaluator feedback.
- **Deterministic validation:** final actions are checked for numerical feasibility before EnergyPlus actuation.
- **Fallback control:** invalid or unusable actions revert to the deterministic occupancy-based rule.

Two LLM implementations are included:

- **DeepSeek Agentic**
- **Kimi K3 Agentic**

The framework is compared with three deterministic reference controllers:

- Fixed schedule
- Occupancy-based rule control
- Comfort/RH-based rule control

---

## ⚙️ Installation

```bash
# Create virtual environment
conda create -n agentic_hvac python=3.10
conda activate agentic_hvac

# Install Python dependency
pip install pandas
```

The experiments require **EnergyPlus 24.1**.

The EnergyPlus Python API (`pyenergyplus`) is included with the EnergyPlus installation.

For LLM experiments, configure the required API key:

```bash
# DeepSeek
set DEEPSEEK_API_KEY=your_key_here

# Kimi / Moonshot AI
set KIMI_API_KEY=your_key_here
```

---

## 🗂 Folder Structure

```text
01_prepare_energyplus_model.py
02_preflight_energyplus_api.py
03_run_fixed_baseline.py
04_run_occupancy_rule_baseline.py
05_run_comfort_rule_baseline.py
06_run_deepseek_agentic.py
07_run_kimi_k3_agentic.py

actual_occupancy_count_5min_7day.csv
actual_occupancy_fraction_energyplus_7day.csv
actual_occupancy_fraction_energyplus_annual.csv

CHN_Hebei.Shijiazhuang.536980_CSWD.epw
CHN_Hebei.Shijiazhuang.536980_CSWD.stat

honeycomb_7zone_fcu_control_v4_experimental.idf

AgenticControlFrameWork.png
```

The scripts automatically create `generated/` and `results/` folders during execution.

---

## 🚀 Usage Examples

1. Prepare the EnergyPlus model

```bash
python 01_prepare_energyplus_model.py
```

2. Run the EnergyPlus API preflight

```bash
python 02_preflight_energyplus_api.py
```

3. Run the deterministic baselines

```bash
python 03_run_fixed_baseline.py
python 04_run_occupancy_rule_baseline.py
python 05_run_comfort_rule_baseline.py
```

4. Run the DeepSeek agentic controller

```bash
python 06_run_deepseek_agentic.py
```

5. Run the Kimi K3 agentic controller

```bash
python 07_run_kimi_k3_agentic.py
```

---

## 🔬 Experimental Setup

- **EnergyPlus version:** 24.1
- **Simulation period:** 16–22 August 2021
- **Control interval:** 5 minutes
- **Controlled zones:** 1, 2, 4, 5, 6, and 7
- **Weather:** Shijiazhuang CSWD
- **Occupancy:** measured 5-minute zone-level occupancy counts
- **LLM control window:** 08:00–18:00
- **Maximum refinement rounds:** 1

---

## 🔬 References

- Citation will be available after publication.
