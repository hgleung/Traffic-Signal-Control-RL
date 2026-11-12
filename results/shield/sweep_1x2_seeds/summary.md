| Scenario | Method | Limit | Avg travel time (s) | Longest red, mean (s) | Longest red, worst (s) | Longest stop, mean (s) | 95th pct. stopped (s) | Episodes |
|---|---|---|---|---|---|---|---|---|
| medium | ppo (as trained) | - | 67.5 ± 0.6 | 298 | 417 | 298 | 54 | 15 |
| high | ppo (as trained) | - | 81.5 ± 1.4 | 364 | 462 | 364 | 99 | 15 |
| medium | ppo + cap | 75 | 69.0 ± 1.0 | 97 | 102 | 97 | 59 | 15 |
| high | ppo + cap | 75 | 91.1 ± 3.2 | 102 | 107 | 120 | 100 | 15 |
| medium | ppo + cap | 90 | 68.5 ± 0.7 | 111 | 121 | 111 | 58 | 15 |
| high | ppo + cap | 90 | 87.6 ± 2.8 | 114 | 123 | 114 | 98 | 15 |
| medium | ppo + cap | 100 | 68.3 ± 0.7 | 120 | 129 | 120 | 57 | 15 |
| high | ppo + cap | 100 | 84.7 ± 1.2 | 126 | 133 | 126 | 96 | 15 |
| medium | ppo + cap | 120 | 68.3 ± 0.9 | 138 | 140 | 138 | 58 | 15 |
| high | ppo + cap | 120 | 83.7 ± 1.5 | 142 | 151 | 142 | 97 | 15 |
| medium | ppo + deadline shield | 75 | 70.0 ± 0.8 | 75 | 75 | 75 | 61 | 15 |
| high | ppo + deadline shield | 75 | 99.0 ± 6.2 | 75 | 75 | 120 | 119 | 15 |
| medium | ppo + deadline shield | 80 | 69.7 ± 1.0 | 80 | 80 | 80 | 61 | 15 |
| high | ppo + deadline shield | 80 | 94.5 ± 5.3 | 80 | 80 | 101 | 106 | 15 |
| medium | ppo + deadline shield | 90 | 69.1 ± 1.0 | 90 | 90 | 90 | 60 | 15 |
| high | ppo + deadline shield | 90 | 90.4 ± 3.7 | 90 | 90 | 109 | 96 | 15 |
| medium | ppo + deadline shield | 100 | 68.8 ± 0.7 | 100 | 100 | 100 | 59 | 15 |
| high | ppo + deadline shield | 100 | 86.9 ± 2.1 | 100 | 100 | 102 | 94 | 15 |
| medium | ppo + deadline shield | 120 | 68.5 ± 1.1 | 120 | 120 | 120 | 58 | 15 |
| high | ppo + deadline shield | 120 | 84.2 ± 1.4 | 120 | 120 | 120 | 94 | 15 |
| medium | actuated | 30 | 77.3 ± 1.0 | 79 | 88 | 79 | 66 | 5 |
| high | actuated | 30 | 119.7 ± 15.8 | 97 | 105 | 176 | 222 | 5 |
| medium | actuated | 40 | 77.7 ± 0.9 | 87 | 95 | 87 | 69 | 5 |
| high | actuated | 40 | 96.0 ± 3.9 | 110 | 117 | 111 | 116 | 5 |
| medium | actuated | 60 | 77.5 ± 0.8 | 91 | 98 | 91 | 68 | 5 |
| high | actuated | 60 | 95.0 ± 2.4 | 142 | 144 | 142 | 111 | 5 |
