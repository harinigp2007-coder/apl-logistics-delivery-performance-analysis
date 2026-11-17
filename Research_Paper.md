
# Delivery Performance, Delay Risk, and Logistics Efficiency
## Analysis in Global Supply Chain Operations

### 1. Introduction

This project analyzes delivery performance and logistics efficiency
using the APL Logistics supply chain dataset.

### 2. Problem Statement

Delivery delays can affect customer satisfaction, shipping costs,
and operational efficiency. This analysis evaluates delivery
performance, shipping modes, geographical regions, and customer
segments to identify potential logistics bottlenecks.

### 3. Dataset Description

Original dataset records: 180519

Eligible records used for delivery analysis: 163472

The analysis excludes cancelled shipments and records with
scheduled shipping duration equal to zero.

### 4. Methodology

- Data cleaning and validation
- Delivery delay gap calculation
- Delivery performance classification
- Late delivery risk analysis
- Shipping mode comparison
- Market, region and country analysis
- Customer segment analysis
- KPI calculation and visualization

### 5. Key Performance Indicators

- On-Time Delivery Rate: 42.18%
- Average Delivery Delay Gap: 0.57 days
- Late Delivery Risk Ratio: 57.82%
- Total Eligible Shipments: 163472

On-time delivery rate includes both early and exactly on-time
shipments.

### 6. Delivery Performance

Delayed shipments: 94523
Early shipments: 41592
On-time shipments: 27357

### 7. Shipping Mode Analysis

The shipping mode with the highest average delay gap is:
Second Class.

### 8. Regional Analysis

The region with the highest average delivery delay gap is:
Central Asia.

### 9. Country Analysis

The country with the highest average delivery delay gap is:
Luxemburgo.

### 10. Customer Segment Analysis

The customer segment with the highest average delivery delay gap is:
Home Office.

### 11. Limitations

The late_delivery_risk field is an existing dataset label.
It matches the delivery status classification in this analysis.
Therefore, it is treated as an existing risk indicator and not
as an independently validated predictive model.

The analysis is observational and does not establish the causes
of delivery delays.

### 12. Recommendations

- Monitor shipping modes with higher average delay gaps.
- Review operational bottlenecks in high-delay regions.
- Monitor country-level delivery performance.
- Compare delivery experience across customer segments.
- Use dashboard filters for continuous performance monitoring.

### 13. Conclusion

This project provides a data-driven overview of delivery
performance, delay risk, shipping mode efficiency, geographical
patterns, and customer segment impact.

The findings can support logistics teams in identifying
areas that require further operational investigation.
