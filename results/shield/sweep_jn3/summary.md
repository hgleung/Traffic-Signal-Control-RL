| Scenario | Method | Limit | Avg travel time (s) | Longest red, mean (s) | Longest red, worst (s) | Longest stop, mean (s) | 95th pct. stopped (s) | Episodes |
|---|---|---|---|---|---|---|---|---|
| real | ppo (as trained) | - | 295.0 ± 1.4 | 747 | 950 | 747 | 166 | 15 |
| 2000 | ppo (as trained) | - | 281.7 ± 0.8 | 415 | 594 | 415 | 105 | 15 |
| 2500 | ppo (as trained) | - | 277.1 ± 1.2 | 525 | 844 | 525 | 120 | 15 |
| real | ppo + cap | 90 | 303.0 ± 2.2 | 119 | 123 | 121 | 183 | 15 |
| 2000 | ppo + cap | 90 | 282.5 ± 0.9 | 111 | 119 | 111 | 108 | 15 |
| 2500 | ppo + cap | 90 | 279.3 ± 1.4 | 114 | 121 | 114 | 122 | 15 |
| real | ppo + cap | 120 | 300.0 ± 2.3 | 146 | 151 | 146 | 178 | 15 |
| 2000 | ppo + cap | 120 | 282.1 ± 1.0 | 140 | 149 | 140 | 107 | 15 |
| 2500 | ppo + cap | 120 | 277.9 ± 1.3 | 142 | 151 | 142 | 124 | 15 |
| real | ppo + deadline shield | 75 | 312.4 ± 4.1 | 75 | 75 | 77 | 212 | 15 |
| 2000 | ppo + deadline shield | 75 | 284.8 ± 0.9 | 75 | 75 | 75 | 107 | 15 |
| 2500 | ppo + deadline shield | 75 | 282.7 ± 1.4 | 75 | 75 | 75 | 125 | 15 |
| real | ppo + deadline shield | 90 | 307.2 ± 3.6 | 90 | 90 | 96 | 194 | 15 |
| 2000 | ppo + deadline shield | 90 | 283.7 ± 0.7 | 90 | 90 | 90 | 108 | 15 |
| 2500 | ppo + deadline shield | 90 | 280.8 ± 1.5 | 90 | 90 | 90 | 122 | 15 |
| real | ppo + deadline shield | 120 | 301.5 ± 2.2 | 120 | 120 | 120 | 179 | 15 |
| 2000 | ppo + deadline shield | 120 | 282.4 ± 0.9 | 120 | 120 | 120 | 108 | 15 |
| 2500 | ppo + deadline shield | 120 | 279.0 ± 1.3 | 120 | 120 | 120 | 124 | 15 |
