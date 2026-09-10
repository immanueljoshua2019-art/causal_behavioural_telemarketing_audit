# A Behavioral-Economics Read of Bank Marketing Data
### *Bridging Predictive Machine Learning and Causal Identification with Bagging and Boosting*[cite: 1]

**Joshua Immanuel**[cite: 1]  
*Decision Science • Causal Machine Learning • Applied Micro-Econometrics • Behavioral Economics*[cite: 1]

[](#)
[](#)
[](#)
[](LICENSE)
[](#)

---



## Executive Abstract

In direct marketing and algorithmic decision systems, teams routinely commit a fundamental error: **they treat predictive feature importance as a mandate for causal policy intervention.**

When an ensemble tree model flags contact frequency or macroeconomic sentiment as a high-leverage feature, standard business intelligence assumes that altering those inputs will mechanically improve conversion. In reality, observational customer logs are heavily confounded by targeting heuristics, unobserved client intent, and temporal clustering.

In this project, I conduct an end-to-end empirical audit of $41,188$ telemarketing contacts from the UCI Bank Marketing dataset[cite: 1]. I pit non-parametric predictive ensembles (**Random Forest** and **HistGradientBoosting**) against formal structural econometrics (**Pearl’s Backdoor Criterion**, **Inverse Probability of Treatment Weighting [IPTW]**, **Cluster-Robust Covariance Corrections**, and **Falsification Placebos**)[cite: 1].

The analysis investigates four core tensions between behavioral economics and neoclassical financial theory, proving where predictive correlations hold up under counterfactual scrutiny—and where they catastrophically collapse[cite: 1].

```text
  ┌─────────────────────────────────────────┐         ┌─────────────────────────────────────────┐
  │      PREDICTIVE MACHINE LEARNING        │         │      STRUCTURAL CAUSAL INFERENCE        │
  ├─────────────────────────────────────────┤         ├─────────────────────────────────────────┤
  │ • Bagging: Random Forest (300 Trees)    │         │ • Directed Acyclic Graphs (DAGs)        │
  │ • Boosting: HistGradientBoosting (300)  │         │ • Inverse Probability Weighting (IPTW)  │
  │ • Metric: PR-AUC (Rare Positive Focus)  │   ───▶  │ • Backdoor Covariate Sets (Z: 48-51 vars)│
  │ • Harmonized OOS Permutation Drop (AUC) │         │ • Cluster-Robust WLS (375 Macro Blocks) │
  │ • Non-Parametric Marginals (PDP / ICE)  │         │ • Treatment Permutation Placebos        │
  │ • Isotonic Risk Calibration (0.0737)    │         │ • Negative Control Outcomes & E-Values  │
  └─────────────────────────────────────────┘         └─────────────────────────────────────────┘
```

---

## Key Empirical Findings & Cross-Question Synthesis

| Dimension | Q1: Contact Cadence (Reactance vs. Mere-Exposure)[cite: 1] | Q2: Past Outreach (Reinforcement vs. Fatigue)[cite: 1] | Q3: Active Debt (Mental Accounting vs. Liquidity)[cite: 1] | Q4: Macro Sentiment (Animal Spirits vs. Fundamentals)[cite: 1] |
|:---|:---|:---|:---|:---|
| **Competing Theories** | Psychological Reactance (Brehm) vs. Mere-Exposure (Zajonc)[cite: 1] | Reinforcement / Availability vs. Habituated Refusal[cite: 1] | Debt Aversion (Prelec-Loewenstein) vs. Cash-Flow Constraints[cite: 1] | "Animal Spirits" (Akerlof-Shiller) vs. Rational Neoclassical Rates[cite: 1] |
| **Predictive ML Behavior** | Strictly monotonic negative slope across PDP/ICE grids in both models[cite: 1]. | `poutcome_success` dominates top split tiers across RF and GB[cite: 1]. | Near-zero permutation importance ($\Delta\text{AUC} \le 0.0003$); flat PDPs[cite: 1]. | Moderate out-of-sample perm importance ($\text{GB }\Delta\text{AUC} = +0.0063$, $\sim 4.8\times \text{Std}$)[cite: 1]. |
| **Causal Identification Design** | IPTW backdoor conditioning; threshold-step audit ($\ge 3, \ge 4, \ge 5$)[cite: 1]. | Natural conditional stratification on historical outcome states[cite: 1]. | IPTW conditioning on 48 backdoor capacity, demographic, and macro covariates[cite: 1]. | Backdoor IPTW on fundamentals + WLS with Cluster-Robust SEs across 375 macro blocks[cite: 1]. |
| **Estimated Causal Effect ($\text{ATE}$)** | **$-1.82\text{ pp}$** at $\ge 4$ calls ($p = 3.4 \times 10^{-9}$); **$-3.16\text{ pp}$** at $\ge 5$ calls[cite: 1]. | **$+56.3\text{ pp}$** raw surge for prior success ($65.1\%$ vs. $8.8\%$ cold baseline)[cite: 1]. | **$-0.30\text{ pp}$** ($p = 0.3402$, $95\%\text{ CI } [-0.90\text{ pp}, +0.31\text{ pp}]$)[cite: 1]. | **$+2.51\text{ pp}$** ($p = 0.1413$, $95\%\text{ CI } [-0.83\text{ pp}, +5.85\text{ pp}]$)[cite: 1]. |
| **Falsification Status** | Placebo refutation passed ($p > 0.50$, effect collapsed to zero)[cite: 1]. | Triangulated across parametric and non-parametric sub-cohort checks[cite: 1]. | Placebo refutation passed ($+0.11\text{ pp}$, $p = 0.7382$)[cite: 1]. | Cluster placebo passed ($p = 0.2622$); Predetermined `age` control passed ($p = 0.3059$)[cite: 1]. |
| **Final Decision Verdict** | **Confirmed Causal Law:** Causal harm begins abruptly at call #4[cite: 1]. | **Strong Empirical Regularity:** Dominant carryover; rejections remain viable ($14.2\%$)[cite: 1]. | **Empirically Refuted:** Active debt exerts zero drag on term deposit adoption[cite: 1]. | **Empirically Refuted:** Sentiment carries zero independent causal power net of fundamentals[cite: 1]. |

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

## Methodological Integrity & AI Collaboration Disclosure

In keeping with modern scientific best practices, I disclose my workflow toolchain:

> "I leverage generative AI models as a technical compiler, boilerplate syntax accelerator, and diagnostic sounding board.
> 
> However, the methodological architecture—identifying target leakage in call duration[cite: 1], resolving the metric mismatch between MDI and Permutation Importance[cite: 1], detecting pseudo-replication across the 375 macroeconomic episodes[cite: 1], specifying the structural DAGs[cite: 1], and diagnosing the Brier score distortion under balanced loss weights—was conceived, directed, and audited entirely by myself as lead researcher."

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



**Lead Researcher & Author:** **Joshua Immanuel**[cite: 1]  
*Decision Science • Causal Machine Learning • Applied Micro-Econometrics • Behavioral Economics*[cite: 1]
