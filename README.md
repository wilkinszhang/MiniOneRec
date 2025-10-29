<div align="center">

# MiniOneRec

**An Open-Source Framework for
Scaling Generative Recommendation**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)

</div>

MiniOneRec is the first fully open-source **generative recommendation** framework, which provides an end-to-end workflow spanning **SID construction**, **supervised fine-tuning (SFT)**, and recommendation-oriented **reinforcement learning (RL)**. 

---

## 🗂️ Repository Overview

| File / Directory          | Description                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `sft.sh`                  | Shell script that launches the Supervised Fine-Tuning (SFT) stage                                             |
| `sft.py`                  | Python implementation of the SFT training loop                                                                |
| `rl.sh`                   | Shell script that launches the Reinforcement Learning (RL) stage                                      |
| `rl.py`                   | Python implementation of the RL training loop                                                         |
| `minionerec_trainer.py`   | **MiniOneRec trainer** – GRPO trainer tailored for generative recommendation                                 |
| `configs/`                | YAML configuration files (datasets, backbone, hyper-parameters)                                              |
| `evaluate.sh`     | One-click offline top-K evaluation script                                                                    |
| `evaluate.py`     | Evaluation code that computes HR@K, NDCG@K, etc.                                                              |
| `requirements.txt`        | Python dependency list                                                                                        |

---

## 🚀 Quickstart

### 1. Create a virtual environment

```bash
conda create -n MiniOneRec python=3.9 -y
conda activate MiniOneRec
```

### 2. Install required packages.

```bash
pip install -r requirements.txt
```

### 3. SFT.

```bash
bash sft.sh
```

### 4. RL.

```bash
bash rl.sh
```

### 5. Run the evaluation bash.

```bash
bash evaluate.sh
```
