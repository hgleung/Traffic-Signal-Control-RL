| Scenario | Method | Limit | Avg travel time (s) | Longest red, mean (s) | Longest red, worst (s) | Longest stop, mean (s) | 95th pct. stopped (s) | Episodes |
|---|---|---|---|---|---|---|---|---|
| medium | dsppo75 (as trained) | - | 70.4 ± 0.9 | 75 | 75 | 75 | 57 | 15 |
| high | dsppo75 (as trained) | - | 102.1 ± 12.6 | 75 | 75 | 176 | 143 | 15 |
| medium | dsppo75 + deadline shield | 75 | 70.4 ± 0.9 | 75 | 75 | 75 | 57 | 15 |
| high | dsppo75 + deadline shield | 75 | 102.1 ± 12.6 | 75 | 75 | 176 | 143 | 15 |
| medium | dsppo75 + deadline shield | 90 | 70.1 ± 1.1 | 89 | 90 | 89 | 59 | 15 |
| high | dsppo75 + deadline shield | 90 | 90.9 ± 4.7 | 90 | 90 | 115 | 102 | 15 |
