# AI and ML for Cybersecurity – Final Exam

Author: Goga Surguladze  
Course: AI and ML for Cybersecurity  
Date: February 12, 2026  

---

## Repository Structure

This repository contains three tasks:

```
task_1/  → Convolutional Neural Network (CNN)
task_2/  → Transformer Network
task_3/  → DDoS Detection using Regression Analysis
```

---

## Task 1 – Convolutional Neural Network

File:
- task_1/conv_net.md

Contains:
- CNN architecture explanation
- Mathematical formulation of convolution
- Cybersecurity application (Malware Image Classification)
- PyTorch implementation
- Feature map visualization

---

## Task 2 – Transformer Network

File:
- task_2/transformer.md

Contains:
- Self-attention mechanism explanation
- Mathematical formulation
- Attention heatmap visualization
- Positional encoding explanation and visualization
- Cybersecurity applications

---

## Task 3 – Web Server Log Analysis

Folder:
- task_3/

Contains:
- ddos.md → Full regression analysis report
- ddos_regression.py → Source code
- server.log → Provided log file
- ddos_plot.png → Traffic visualization

### Identified DDoS Interval

The DDoS attack was detected between:

**2024-03-22 18:37 – 18:41 (+04:00)**

Detection was performed using linear regression and residual threshold analysis.

---

## Reproducibility

To reproduce Task 3:

1. Install required libraries:
   - pandas
   - matplotlib
   - scikit-learn
   - numpy

2. Run:
   ```
   python ddos_regression.py
   ```

---

End of repository.
