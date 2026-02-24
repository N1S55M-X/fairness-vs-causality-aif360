# ⚖️ Fairness vs Causality: AIF360 Bias Mitigation Pipeline

## 📌 Overview

This repository implements a complete **algorithmic fairness auditing and mitigation pipeline** using **IBM’s AI Fairness 360 (AIF360)** toolkit.

The project demonstrates:

- Bias detection using fairness metrics  
- Pre-processing mitigation (Reweighing + Disparate Impact Remover)  
- In-processing mitigation (Adversarial Debiasing)  
- Post-processing mitigation (Equalized Odds)  
- Fairness vs. accuracy trade-off analysis  

Although applied to the Concrete Compressive Strength dataset, this project serves as a methodological demonstration of **Responsible AI workflows**.

---

## 🏢 Why IBM AIF360?

This project uses **IBM’s AI Fairness 360 (AIF360)** toolkit because it is one of the most widely recognized open-source libraries for bias detection and mitigation in machine learning systems.

AIF360 is commonly used in:

- Industry Responsible AI pipelines  
- Regulatory compliance workflows  
- Academic fairness research  
- Enterprise AI governance frameworks  

It provides:

- Standardized fairness metrics (Disparate Impact, Statistical Parity Difference, Equal Opportunity)  
- Pre-processing, in-processing, and post-processing mitigation algorithms  
- Reproducible and auditable fairness evaluation tools  

Using AIF360 ensures that:

- Fairness evaluation follows recognized statistical definitions  
- Results align with regulatory standards such as the **80% rule (Disparate Impact ≥ 0.8)**  
- Mitigation methods reflect real-world Responsible AI practices  

Rather than implementing custom fairness logic, this project leverages a trusted, research-backed framework to demonstrate structured and industry-aligned bias auditing.

---

## 🧠 Motivation

Machine learning models can achieve high accuracy while still producing **statistically imbalanced outcomes across subgroups**.

This project explores how fairness mitigation techniques impact predictive performance when applied to structured engineering data.

---

## 📉 Trade-Off Summary

| Stage | Disparate Impact | SPD | EOD | Accuracy |
|--------|------------------|-----|-----|----------|
| After In-Processing | 0.323 | -0.629 | -0.204 | 0.806 |
| After Equalized Odds | 0.916 | -0.074 | 0.029 | 0.602 |

### Interpretation

- Fairness metrics significantly improved after Equalized Odds (DI > 0.8 threshold).
- Statistical Parity Difference moved close to zero.
- Accuracy decreased substantially.

This demonstrates the classic:

> ⚖️ Fairness–Accuracy Trade-off

---

## 🎯 Key Insight

When fairness mitigation is applied to datasets where group differences reflect genuine underlying structure, enforcing statistical parity may reduce predictive performance.

This repository demonstrates:

- How fairness toolchains operate  
- How mitigation affects model behavior  
- Why domain context matters in fairness auditing  

---

## 💼 Real-World Use Cases

Fairness auditing and mitigation techniques are widely applied in:

- 🏦 Loan approval systems  
- 🏥 Healthcare risk prediction  
- 👩‍⚖️ Criminal justice risk scoring  
- 🏢 Hiring and recruitment models  
- 🎓 Education admissions systems  

---

**Note:** The protected attribute in this experiment is simulated for methodological demonstration purposes.
