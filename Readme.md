
# Fear Generalization 

This repository contains all the necessary files, code, and outputs related to the Fear Generalization project. It includes model implementation, analysis, simulation, and results from both R and Python environments.

---

## Project Overview
This project investigates fear generalization through computational modeling. The report includes a detailed explanation of the model and its implementation in both R and Python. The primary goal is to analyze fear responses and generalization patterns using experimental data.

---

## Contents of the Repository

### **1. Report**
- The full explanation of the model and its underlying code can be found in the **`Fear_Generalization_report.pdf`** file.

---

### **2. Code**

#### **R Code**
- The R code for the project, sourced from the repository [OSF](https://osf.io/sxjak/), is located in the `r` folder. This folder includes:
  - R scripts for data manipulation and analysis.
  - **JAGS** code used for sampling and Bayesian model fitting.

#### **Python Code**
- The Python implementation of the model (using **PyMC**) and data processing scripts are located in the `python` folder. 
- Structure of the `python` folder:
  - **`1_data_clean/`**: Scripts for cleaning and preparing the data for analysis.
  - **`2_analysis/`**: Scripts for running analyses and model fitting.
  - **`3_simulation/`**: Scripts for simulating data and generating hypothesized participant behaviors.

- Files are available in both `.py` format (executable scripts) and `.ipynb` notebooks, which include step-by-step explanations of the code.

---

### **3. Python Requirements**
To run the Python code, the following libraries must be installed:
- `pandas`
- `numpy`
- `pickle`
- `natsort`
- `matplotlib`
- `seaborn`
- `pymc`

Use the following command to install the required libraries:
```bash
pip install pandas numpy pickle natsort matplotlib seaborn pymc
```

---

### **4. Plots**
- All plots generated through the Python scripts are saved in the `Plots/py` folder. 
- Subfolders:
  - **`dataclean/`**: Plots generated during the data cleaning process.
  - **`simulation/`**: Plots illustrating simulated results based on hypothesized models.
  - **`hypothesizedParticipant/`**: Plots showing hypothesized participant responses.

**Note:** Ensure the directory structure is in place before executing the Python scripts.

---

### **5. Data**
- The `Data/` folder contains the experimental data and intermediate results:
  - **`Experiment_1/`** and **`Experiment_2/`**: Original experimental data.
  - **`fitting_res_py/`** and **`fitting_res_r/`**: Results from sampling models in Python and R, respectively.
  - **`res_py/`** and **`res_r/`**: Processed and manipulated data outputs in Python and R, respectively.

---

## Usage Instructions
1. Ensure you have the required Python libraries installed (see the Python Requirements section).
2. Navigate to the appropriate folder (e.g., `python/2_analysis`) and execute the desired `.py` or `.ipynb` file.
3. Generated plots will be automatically saved in the `Plots/py` folder.
4. Refer to the **`Fear_Generalization_report.pdf`** for detailed information on the model, methodology, and findings.

---

## Acknowledgments
This project utilizes code from the [OSF Repository](https://osf.io/sxjak/) for R-based analysis and JAGS modeling.

---
