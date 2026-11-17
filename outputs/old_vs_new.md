| Metric | Class report | Class pipeline rerun | New, test set | Why |
|---|---|---|---|---|
| ROC-AUC | 0.946 | 0.948 | 0.553 | Amount fill value, Qty and promotion leaks removed; time split |
| PR-AUC | n/a | 0.858 | 0.168 | Not reported in class version |
| Sensitivity | 0.839 | 0.836 | 0.000 | Class used a 0.5 cutoff; new uses a validation cutoff from the 3x cost assumption |
| Precision | n/a | 0.561 | n/a | Not reported in class version |
| Threshold | 0.500 | 0.500 | flag none | Chosen on validation; at 3x no cutoff beat flagging nothing |
| Sensitivity at 5x cost cutoff | n/a | n/a | 0.118 | Secondary cost assumption, cutoff from validation |
| Precision at 5x cost cutoff | n/a | n/a | 0.186 | Secondary cost assumption, cutoff from validation |
| Positive rate | 0.151 | 0.151 | 0.146 | Status Cancelled only, open orders dropped |
