<div align="center">


<img src="./assets/logo.png" width="500em" ></img> 

**An Open-Source Framework for
Scaling Generative Recommendation**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)

</div>

MiniOneRec is the first fully open-source **generative recommendation** framework, which provides an end-to-end workflow spanning **SID construction**, **supervised fine-tuning (SFT)**, and recommendation-oriented **reinforcement learning (RL)**. 

---

## Key Techniques 
<div align="center">
<img src="./assets/minionerec_framework.png" width=100% ></img> 
</div>

- **SID Construction: MiniOneRec begins by transforming every product into a compact, semantically meaningful token.** It concatenates an item’s title and description, feeds this sentence through a frozen text encoder, and then quantises the resulting embedding with a three-level RQ-VAE. The outcome is a three-byte SID that both preserves hierarchical semantics and keeps the vocabulary extremely small.

- **SFT: With all items rewritten as SIDs, the model is first trained in a supervised fashion.** It views the chronologically ordered user history as a token sequence and learns, via next-token prediction, to generate the SID of the next product the user is likely to consume. Crucially, this stage is co-trained with a set of language-alignment objectives that map back and forth between natural language and SID space, allowing the recommender to inherit the world knowledge embedded in large language models while grounding that knowledge in discrete item codes.

- **Recommendation-Oriented RL: After SFT, MiniOneRec is further polished with a recommendation-oriented RL phase based on GRPO.** Multiple candidate recommendations are generated for each prompt, their rewards are normalised within the group to stabilise gradients, and a KL penalty keeps the updated policy close to its reference. Because the action space is a closed list of item SIDs, the system switches to constrained beam search, which guarantees that every beam is unique and valid, greatly improving sampling efficiency and diversity. The reward signal itself blends a binary correctness term with a rank-aware component that penalises high-probability yet incorrect items more heavily, and can be augmented with collaborative-filtering scores. Together, this pipeline enables MiniOneRec to couple dense linguistic knowledge, achieving a high-performance, lightweight generative recommendation system.


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
