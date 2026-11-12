| Scenario | Method | Limit | Avg travel time (s) | Longest red, mean (s) | Longest red, worst (s) | Longest stop, mean (s) | 95th pct. stopped (s) | Episodes |
|---|---|---|---|---|---|---|---|---|
| real | ppo (as trained) | - | 316.0 ± 1.9 | 359 | 537 | 359 | 66 | 15 |
| peak | ppo (as trained) | - | 388.1 ± 2.1 | 1401 | 2044 | 1401 | 254 | 15 |
| real | ppo + cap | 90 | 316.3 ± 1.8 | 109 | 117 | 109 | 68 | 15 |
| peak | ppo + cap | 90 | 397.8 ± 3.0 | 115 | 123 | 115 | 243 | 15 |
| real | ppo + cap | 120 | 316.2 ± 1.8 | 137 | 139 | 137 | 67 | 15 |
| peak | ppo + cap | 120 | 394.5 ± 3.0 | 143 | 151 | 143 | 241 | 15 |
| real | ppo + deadline shield | 75 | 316.5 ± 1.7 | 75 | 75 | 75 | 68 | 15 |
| peak | ppo + deadline shield | 75 | 402.4 ± 2.0 | 75 | 75 | 75 | 246 | 15 |
| real | ppo + deadline shield | 90 | 316.5 ± 1.6 | 90 | 90 | 90 | 68 | 15 |
| peak | ppo + deadline shield | 90 | 399.6 ± 3.0 | 90 | 90 | 90 | 245 | 15 |
| real | ppo + deadline shield | 120 | 316.2 ± 1.9 | 120 | 120 | 120 | 67 | 15 |
| peak | ppo + deadline shield | 120 | 396.4 ± 2.3 | 120 | 120 | 120 | 246 | 15 |
