| Policy | Fields automated | Accuracy of automated fields | Accuracy if reviewed fields are fixed | Cost per 1,000 receipts | Mean latency (s) |
| --- | --- | --- | --- | --- | --- |
| Tier 1 only (gpt-4o-mini, 3 runs) | 100.0% | 66.3% | 66.3% | $0.30 | 2.1 |
| Tier 2 only (gpt-4o) | 100.0% | 89.7% | 89.7% | $3.34 | 1.9 |
| Tier 2 only, gated at the same threshold (0.66) | 90.3% | 94.5% | 95.0% | $3.34 | 1.9 |
| Cascade (threshold 0.66) | 92.7% | 84.2% | 85.3% | $2.61 | 3.5 |
