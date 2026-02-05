# 🧠 EEG-Brain-DA

![EEG](https://img.shields.io/badge/EEG-Deep%20Learning-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**EEG-Brain-DA** is a research project focused on deep learning analysis of EEG (electroencephalogram) data for neurological assessment and prognosis prediction. This project explores preprocessing pipelines, baseline models, and foundation models for EEG signal analysis.

---

## 📑 Table of Contents

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Development Team](#development-team)
- [Getting Started](#getting-started)
  - [Python Environment Setup](#python-environment-setup)
  - [Install Dependencies](#install-dependencies)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [How to Contribute](#how-to-contribute)
- [License](#license)

---

## 📘 About the Project

- **Objective:**  
  Develop and evaluate deep learning models for EEG signal analysis, focusing on neurological prognosis prediction and brain activity pattern recognition.

- **Approach:**  
  Implementation of preprocessing pipelines, baseline models (LSTM, CNN), and exploration of foundation models (Transformers, BERT-based architectures) for time-series EEG data.

- **Institution:**  
  Escuela Tecnológica Instituto Técnico Central (ETITC)

---

## 🚀 Key Features

- End-to-end EEG preprocessing and feature extraction pipelines
- Baseline deep learning models (LSTM, CNN, InceptionTime)
- Foundation model adaptation for EEG time-series data
- Demographic and clinical data integration
- Reproducible Jupyter notebook workflows
- Modular utility functions for training and evaluation

---

## 👨👩👧👦 Development Team

| Name                  | GitHub Username                              |
|-----------------------|----------------------------------------------|
| Francisco Gomez       | -                                            |
| Juan Leal             | -                                            |
| Baran                 | -                                            |
| Sebastián Yepes       | [@byepesg](https://github.com/byepesg)       |
| Manuel Pacheco        | -                                            |

---

## ⚙️ Getting Started

### Python Environment Setup

We recommend using a virtual environment to keep dependencies isolated:

```bash
# 1. Clone the repository
git clone git@github.com:EEG-ETITC/EEG-Brain-DA.git
cd EEG-Brain-DA

# 2. Create virtual environment
python -m venv venv_cerebro

# 3. Activate it
# Windows
venv_cerebro\Scripts\activate
# Linux/macOS
source venv_cerebro/bin/activate

# 4. Upgrade pip
pip install --upgrade pip
```

### Install Dependencies

```bash
# Install required packages
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn jupyter notebook
pip install mne  # For EEG data processing
```

> **Note:** A `requirements.txt` file will be added in future updates for easier dependency management.

---

## 📁 Project Structure

```
EEG-Brain-DA/
│
├── Code/
│   ├── Utils/                      # Utility modules
│   │   ├── LSTM_net.py            # LSTM network architecture
│   │   ├── download.py            # Data download utilities
│   │   └── train.py               # Training utilities
│   │
│   ├── 01_preprocessing.ipynb      # EEG preprocessing pipeline
│   ├── 02_baseline_models.ipynb    # Baseline model implementations
│   ├── 03_foundation_models.ipynb  # Foundation model experiments
│   ├── demographinc_analisys.ipynb # Demographic data analysis
│   ├── preprocessing_pipeline_nb.ipynb  # Complete preprocessing workflow
│   └── ...                         # Additional notebooks
│
├── MAPI/                           # Medical API and visualization resources
│   └── Esquema_cabeza.png         # EEG electrode placement diagram
│
├── articles/                       # Research papers and references
├── Official Documents/             # Project documentation
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 📓 Notebooks

### Preprocessing & Feature Extraction
- **`01_preprocessing.ipynb`** - EEG signal preprocessing (filtering, artifact removal, normalization)
- **`preprocessing_pipeline_nb.ipynb`** - Complete preprocessing workflow with feature extraction

### Model Development
- **`02_baseline_models.ipynb`** - Implementation of baseline models (LSTM, CNN, InceptionTime)
- **`03_foundation_models.ipynb`** - Exploration of foundation models for EEG analysis
- **`eeg_prediction_bert.ipynb`** - BERT-based architecture for EEG prediction

### Analysis
- **`demographinc_analisys.ipynb`** - Demographic and clinical data analysis
- **`proving.ipynb`** - Model validation and testing

### Data Management
- **`dload_1_ETIC.ipynb`** - Data download and organization for ETITC datasets

---

## 🤝 How to Contribute

We welcome all forms of contributions including research ideas, code, bug reports, and documentation fixes!

### 1. Fork and Clone

Start by forking the repository to your own GitHub account. Then clone it using SSH:

```bash
git clone git@github.com:your-username/EEG-Brain-DA.git
cd EEG-Brain-DA
```

### 2. Create a Branch

Always create a new branch for your work:

```bash
git checkout -b yourusername/feature/branch-name
```

**Branch naming examples:**
- `byepesg/feature/data-extraction`
- `byepesg/fix/preprocessing-bug`
- `byepesg/docs/readme-update`

### 3. Write Clear Commit Messages

```bash
git commit -m "feat: add ICA preprocessing pipeline"
```

**Commit prefixes:**
- `feat:` → for new features
- `fix:` → for bug fixes
- `docs:` → for documentation-only changes
- `refactor:` → for code restructuring
- `test:` → for tests
- `chore:` → for tooling, CI, or other housekeeping

### 4. Push and Create a Pull Request

```bash
git push -u origin yourusername/feature/branch-name
```

Then open a Pull Request on GitHub with a clear description of your changes.

---

## 📊 Data Management

> **Important:** This repository contains **code only**. Large data files (`.pt`, `.csv`, `.mat`, `.hea`) are stored separately and excluded via `.gitignore`.

**Data files are managed through:**
- OneDrive for team collaboration
- Local storage for individual development
- External datasets (e.g., PhysioNet, I-CARE)

---

## 📚 References

- EEG signal processing and analysis techniques
- Deep learning for time-series medical data
- Foundation models for physiological signals
- Clinical neurological assessment protocols

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🔗 Links

- **GitHub Repository:** [EEG-ETITC/EEG-Brain-DA](https://github.com/EEG-ETITC/EEG-Brain-DA)
- **Institution:** [ETITC](https://www.itc.edu.co/)

---

**Made with ❤️ by the EEG-Brain-DA Team**