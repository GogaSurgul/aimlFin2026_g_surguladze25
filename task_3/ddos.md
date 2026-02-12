# DDoS Detection Using Regression Analysis

## 1. Introduction

The objective of this task is to analyze a web server log file and detect potential DDoS attack intervals using regression analysis. DDoS attacks typically manifest as abnormal spikes in traffic volume compared to normal baseline activity.

The provided log file was analyzed to extract timestamps and compute request frequency over time.

Log file:
server.log

---

## 2. Data Extraction

The log file contains HTTP request records with timestamps.  
Each timestamp was extracted using regular expressions and converted into datetime format.

Requests were grouped per minute to compute:

- Number of requests per minute
- Time index for regression modeling

---

## 3. Regression Model

A Linear Regression model was applied to estimate expected traffic trends.

Let:

- X = time index
- y = request count per minute

The regression model estimates normal traffic behavior.

Residuals were computed:

Residual = Actual Traffic − Predicted Traffic

To detect anomalies, a threshold was defined as:

Threshold = mean(residuals) + 3 × std(residuals)

Minutes exceeding this threshold were marked as potential DDoS events.

---

## 4. Visualization

The graph below shows:

- Blue line: Actual web traffic
- Orange line: Regression trend

Significant spikes above the regression line indicate abnormal activity.

---

## 5. Identified DDoS Intervals

The following time intervals were identified as potential DDoS attacks:

- 2024-03-22 18:37:00 (+04:00) – 13,604 requests
- 2024-03-22 18:38:00 (+04:00) – 13,432 requests
- 2024-03-22 18:40:00 (+04:00) – 10,256 requests
- 2024-03-22 18:41:00 (+04:00) – 10,398 requests

Overall DDoS window:

🕒 **2024-03-22 18:37 – 18:41 (+04:00)**

These values significantly exceed expected traffic levels predicted by the regression model.

---

## 6. Conclusion

The regression-based anomaly detection successfully identified abnormal traffic spikes consistent with DDoS attack behavior.

The attack occurred between 18:37 and 18:41 on March 22, 2024.

This approach demonstrates how statistical modeling and residual analysis can effectively detect network anomalies in web server logs.
