<div align="center">

# A Behavioural-Economics Read of Bank Marketing Data
### *Bridging Predictive Machine Learning and Causal Identification with Bagging and Boosting*

**Joshua Immanuel**  
*Decision Science • Causal Machine Learning • Applied Micro-Econometrics • Behavioural Economics*

[![DOI](https://zenodo.org/badge/1364544005.svg)](https://doi.org/10.5281/zenodo.22697970)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Ensemble%20Pipelines-F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white)](#)
[![Econometrics](https://img.shields.io/badge/Econometrics-IPTW%20%7C%20Cluster--Robust%20WLS-8A2BE2.svg?style=flat-square)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed%20Research-success.svg?style=flat-square)](#)

---

</div>

## Executive Abstract

In direct marketing and algorithmic decision systems, teams routinely commit a fundamental error: **they treat predictive feature importance as a mandate for causal policy intervention.**

When an ensemble tree model flags contact frequency or macroeconomic sentiment as a high-leverage feature, standard business intelligence assumes that altering those inputs will mechanically improve conversion. In reality, observational customer logs are heavily confounded by targeting heuristics, unobserved client intent, and temporal clustering.

In this project, I conduct an end-to-end empirical audit of $41,188$ telemarketing contacts from the UCI Bank Marketing dataset. I pit non-parametric predictive ensembles (**Random Forest** and **HistGradientBoosting**) against formal structural econometrics (**Pearl’s Backdoor Criterion**, **Inverse Probability of Treatment Weighting [IPTW]**, **Cluster-Robust Covariance Corrections**, and **Falsification Placebos**).

The analysis investigates four core tensions between behavioral economics and neoclassical financial theory, proving where predictive correlations hold up under counterfactual scrutiny—and where they catastrophically collapse.

<table>
  <thead>
    <tr>
      <th width="48%" align="left">🔮 <b>PREDICTIVE MACHINE LEARNING</b></th>
      <th width="4%" align="center"></th>
      <th width="48%" align="left">⚖️ <b>STRUCTURAL CAUSAL INFERENCE</b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td valign="top">
        <ul>
          <li><b>Bagging Architecture:</b> Random Forest (300 Trees)</li>
          <li><b>Boosting Relay:</b> HistGradientBoosting (300 Iterations)</li>
          <li><b>Discriminatory Metric:</b> PR-AUC (Rare-Event Minority Focus)</li>
          <li><b>Attribution Benchmark:</b> Harmonized OOS Permutation Drop ($\Delta\text{AUC}$)</li>
          <li><b>Response Topography:</b> Non-Parametric Marginals (PDP / Centered-ICE)</li>
          <li><b>Uncertainty Calibration:</b> Isotonic Probability Realignment ($0.0737$)</li>
        </ul>
      </td>
      <td align="center" valign="middle">
        <b>───▶</b>
      </td>
      <td valign="top">
        <ul>
          <li><b>Identification Architecture:</b> Directed Acyclic Graphs (DAGs)</li>
          <li><b>Weighting Pipeline:</b> Inverse Probability Weighting (IPTW)</li>
          <li><b>Backdoor Conditioning:</b> Empirical Covariate Sets ($Z$: 48–51 variables)</li>
          <li><b>Variance Correction:</b> Cluster-Robust WLS (375 Macroeconomic Blocks)</li>
          <li><b>Placebo Falsification:</b> Treatment-Permutation Null Refutations</li>
          <li><b>Sensitivity Diagnostics:</b> Predetermined Negative Controls & E-Values</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

---

## Key Empirical Findings & Cross-Question Synthesis

| Evaluation Dimension | Q1: Contact Cadence<br>*(Reactance vs. Mere-Exposure)* | Q2: Past Outreach<br>*(Reinforcement vs. Fatigue)* | Q3: Active Debt<br>*(Mental Accounting vs. Liquidity)* | Q4: Macro Sentiment<br>*(Animal Spirits vs. Fundamentals)* |
|:---|:---|:---|:---|:---|
| **Competing Behavioural Hypotheses** | **Psychological Reactance (Brehm)**<br>vs. Mere-Exposure (Zajonc) | **Reinforcement & Habituation**<br>vs. Habituated Refusal | **Debt Aversion & Mental Accounting**<br>vs. Cash-Flow Constraints | **"Animal Spirits" (Akerlof & Shiller)**<br>vs. Rational Neoclassical Rates |
| **Predictive Ensemble Behavior** | Monotonic negative slope across PDP/ICE grids in both models. | `poutcome_success` dominates top split tiers across RF and GB. | Near-zero permutation importance ($\Delta\text{AUC} \le 0.0003$); flat PDP curves. | Modest OOS permutation importance ($\text{GB }\Delta\text{AUC} = +0.0063$, $\sim 4.8\times \text{Std}$). |
| **Causal Identification Design** | IPTW backdoor conditioning; threshold sensitivity step audit ($\ge 3, \ge 4, \ge 5$). | Natural conditional stratification on historic outcome states and recency windows. | IPTW conditioning on 48 backdoor capacity, demographic, and macro covariates. | Backdoor IPTW on fundamentals + WLS with Cluster-Robust SEs ($375$ macro blocks). |
| **Estimated Causal Effect ($\text{ATE}$)** | **$-1.82\text{ pp}$** at $\ge 4$ calls ($p = 3.4 \times 10^{-9}$)<br>**$-3.16\text{ pp}$** at $\ge 5$ calls | **$+56.3\text{ pp}$** raw surge for prior success ($65.1\%$ vs. $8.8\%$ cold baseline) | **$-0.30\text{ pp}$** ($p = 0.3402$)<br>$95\%\text{ CI } [-0.90\text{ pp}, +0.31\text{ pp}]$ | **$+2.51\text{ pp}$** ($p = 0.1413$)<br>$95\%\text{ CI } [-0.83\text{ pp}, +5.85\text{ pp}]$ |
| **Falsification & Placebo Status** | Placebo refutation passed<br>($p > 0.50$, effect collapsed to zero). | Triangulated across parametric and non-parametric sub-cohort checks. | Placebo refutation passed<br>($+0.11\text{ pp}$, $p = 0.7382$, statistically zero). | Episode placebo passed ($p = 0.2622$); Predetermined `age` control passed ($p = 0.3059$). |
| **Final Decision Verdict** | 🚨 **Confirmed Causal Law:**<br>Causal penalty initiates sharply at contact attempt #4. | 📈 **Strong Empirical Regularity:**<br>Dominant carryover; past rejections remain viable ($14.2\%$). | ⚖️ **Empirically Refuted:**<br>Active debt exerts zero drag on term deposit adoption. | 📉 **Empirically Refuted:**<br>Consumer sentiment carries zero independent causal power. |
---
## Integrated Strategic CRM Blueprint

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          INTEGRATED CRM DECISION ENGINE                          │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  ▼                                             ▼
   ┌───────────────────────────┐                 ┌───────────────────────────┐
   │ MACRO TIMING FILTER (Q4)  │                 │  LEAD SELECTION POOL (Q3) │
   │ Trigger campaigns on real │                 │ DO NOT disqualify leads   │
   │ interest rate drops & low │                 │ with active mortgages or  │
   │ euribor3m / emp.var.rate. │                 │ personal loans (ATE: null)│
   │ (Ignore sentiment polls)  │                 │                           │
   └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                 │                                             │
                 └───────────────────────┬─────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │        PRIOR INTERACTION ROUTING (Q2)         │
                 │ Prior Success -> High-priority immediate tier │
                 │ Prior Failure -> Standard retargeting (viable)│
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │          CONTACT FREQUENCY CAP (Q1)           │
                 │ Strict ceiling of 3 calls per campaign.       │
                 │ Terminate cadence immediately at 4th call     │
                 │ to prevent -1.82pp to -3.16pp reactance drop  │
                 └───────────────────────────────────────────────┘
```

---

## Repository Structure & File Manifest

```text
causal_behavioural_telemarketing_audit/
│
├── causal_behavioural_telemarketing_audit.ipynb  # End-to-end research notebook (Code, DAGs, Diagnostics)
├── bank_marketing_full.csv                      # Benchmark observational dataset (N = 41,188)
├── requirements.txt                             # Pinned computational dependencies & versions
├── LICENSE                                      # MIT Open Source License
└── README.md                                    # Executive research synthesis & empirical verdicts
```

---

## Quickstart & Reproducibility

### 1. Clone Repository & Initialize Virtual Environment

```bash
git clone https://github.com/immanueljoshua2019-art/causal_behavioural_telemarketing_audit.git
cd causal_behavioural_telemarketing_audit

python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Interactive Execution (JupyterLab)

Launch the interactive environment to step through data hygiene routines, ensemble training, and econometric estimators:

```bash
jupyter lab causal_behavioural_telemarketing_audit.ipynb
```

### 3. Headless Verification via Terminal

Execute and validate the complete empirical pipeline deterministically without opening a browser interface:

```bash
jupyter nbconvert --to notebook --execute causal_behavioural_telemarketing_audit.ipynb --output executed_audit.ipynb
```

---

## Research Toolchain & Engineering Ownership

I treat modern AI tools as an execution accelerator—handling repetitive syntax, automating diagnostic plotting routines, and speeding up pipeline development. 

However, every structural decision and core analytical breakthrough in this project was conceived, designed, and verified directly by me:

* **Formulating the Identification Strategy:** I established the causal frameworks pitting neoclassical price theory against behavioural mechanisms (mere-exposure vs. psychological reactance; mental accounting vs. liquidity constraints) to ask questions standard loss functions overlook.
* **Catching Critical Structural Traps:** I identified the macro pseudo-replication and enforced cluster-robust covariance corrections across the 375 macroeconomic episodes when naive regressions returned spurious $p$-values ($p = 10^{-12}$), and determined where model plateaus represented data sparsity rather than real-world equilibrium.
* **Designing for Operational Utility:** I resolved the optimization trade-off between discriminatory ranking for operational lead assignment and isotonic probability calibration for expected-value balance-sheet forecasting.

AI accelerated the code syntax; the causal architecture, econometric specifications, and domain conclusions are entirely my own work.

---

## Citation & Academic Reference

If this causal identification design, threshold audit, or pseudo-replication methodology assists your research or production systems, please cite the repository:

```bibtex
@software{immanuel2026bankcausal,
  author       = {Immanuel, Joshua},
  title        = {A Behavioral-Economics Read of Bank Marketing: Bridging Predictive Ensembles and Structural Causal Identification},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/immanueljoshua2019-art/causal_behavioural_telemarketing_audit}}
}
```

---



**Lead Researcher & Author:** **Joshua Immanuel**  
*Decision Science • Causal Machine Learning • Applied Micro-Econometrics • Behavioural Economics*
