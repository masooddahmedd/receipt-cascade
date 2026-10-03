| Policy | Fields automated | Accuracy of automated fields | Accuracy if reviewed fields are fixed | Cost per 1,000 receipts | Mean latency (s) |
| --- | --- | --- | --- | --- | --- |
| Tier 1 only (gpt-4o-mini, 3 runs) | 100.0% | 72.7% | 72.7% | $0.31 | 2.0 |
| Tier 2 only (gpt-4o) | 100.0% | 89.7% | 89.7% | $3.34 | 1.9 |
| Tier 2 only, gated at the same threshold (0.55) | 91.7% | 93.8% | 94.3% | $3.34 | 1.9 |
| Cascade (threshold 0.55) | 94.3% | 88.3% | 89.0% | $2.75 | 3.4 |
