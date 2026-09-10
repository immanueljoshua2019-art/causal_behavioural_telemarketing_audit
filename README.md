<div align="center">

# A Behavioural-Economics Read of Bank Marketing Data
### *Bridging Predictive Machine Learning and Causal Identification with Bagging and Boosting*

**Author:** **Joshua Immanuel**  
*Decision Science • Causal Machine Learning • Applied Micro-Econometrics • Behavioural economics*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Ensemble%20Pipelines-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](#)
[![Econometrics](https://img.shields.io/badge/Econometrics-IPTW%20%7C%20Cluster--Robust%20WLS-8A2BE2?style=flat-square)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed%20Research-success?style=flat-square)](#)

---

</div>

## Executive Abstract

In direct marketing and algorithmic decision systems, teams routinely commit a fundamental error: **they treat predictive feature importance as a mandate for causal policy intervention.** 

When an ensemble tree model flags contact frequency or macroeconomic sentiment as a high-leverage feature, standard business intelligence assumes that altering those inputs will mechanically improve conversion. In reality, observational customer logs are heavily confounded by targeting heuristics, unobserved client intent, and temporal clustering.

In this project, I conduct an end-to-end empirical audit of $41,188$ telemarketing contacts from the UCI Bank Marketing dataset. I pit non-parametric predictive ensembles (**Random Forest** and **HistGradientBoosting**) against formal structural econometrics (**Pearl’s Backdoor Criterion**, **Inverse Probability of Treatment Weighting [IPTW]**, **Cluster-Robust Covariance Corrections**, and **Falsification Placebos**). 

The analysis investigates four core tensions between behavioral economics and neoclassical financial theory, proving where predictive correlations hold up under counterfactual scrutiny—and where they catastrophically collapse.

```text
  ┌─────────────────────────────────────────┐         ┌─────────────────────────────────────────┐
  │      PREDICTIVE MACHINE LEARNING        │         │      STRUCTURAL CAUSAL INFERENCE        │
  ├─────────────────────────────────────────┤         ├─────────────────────────────────────────┤
  │ • Bagging: Random Forest (300 Trees)    │         │ • Directed Acyclic Graphs (DAGs)        │
  │ • Boosting: HistGradientBoosting (300)  │         │ • Inverse Probability Weighting (IPTW)  │
  │ • Metric: PR-AUC (Rare Positive Focus)  │   ───▶  │ • Backdoor Covariate Sets (Z: 48-51 vars)│
  │ • Harmonized OOS Permutation Drop (AUC) │         │ • Cluster-Robust WLS (375 Macro Blocks) │
  │ • Non-Parametric Marginals (PDP / ICE)  │         │ • Treatment Permutation Placebos        │
  │ • Isotonic Risk Calibration ($0.0737$)  │         │ • Negative Control Outcomes & E-Values  │
  └─────────────────────────────────────────┘         └─────────────────────────────────────────┘

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
