| Scenario | Method | Limit | Avg travel time (s) | Longest red, mean (s) | Longest red, worst (s) | Longest stop, mean (s) | 95th pct. stopped (s) | Episodes |
|---|---|---|---|---|---|---|---|---|
| real | ppo (as trained) | - | 315.9 ± 2.1 | 339 | 425 | 339 | 67 | 5 |
| peak | ppo (as trained) | - | 388.9 ± 2.5 | 1631 | 2044 | 1631 | 244 | 5 |
| real | ppo + cap | 90 | 316.5 ± 2.0 | 109 | 111 | 109 | 70 | 5 |
| peak | ppo + cap | 90 | 400.9 ± 1.6 | 116 | 119 | 116 | 248 | 5 |
| real | ppo + cap | 120 | 316.1 ± 2.0 | 138 | 139 | 138 | 69 | 5 |
| peak | ppo + cap | 120 | 397.1 ± 1.1 | 141 | 146 | 141 | 242 | 5 |
| real | ppo + deadline shield | 75 | 316.4 ± 1.9 | 75 | 75 | 75 | 69 | 5 |
| peak | ppo + deadline shield | 75 | 404.0 ± 1.0 | 75 | 75 | 75 | 250 | 5 |
| real | ppo + deadline shield | 90 | 316.4 ± 1.9 | 90 | 90 | 90 | 70 | 5 |
| peak | ppo + deadline shield | 90 | 402.1 ± 1.8 | 90 | 90 | 90 | 252 | 5 |
| real | ppo + deadline shield | 120 | 316.2 ± 2.2 | 120 | 120 | 120 | 69 | 5 |
| peak | ppo + deadline shield | 120 | 398.3 ± 2.6 | 120 | 120 | 120 | 243 | 5 |
| real | mp (as trained) | - | 352.9 ± 2.3 | 1296 | 1370 | 1296 | 270 | 5 |
| peak | mp (as trained) | - | 427.6 ± 3.3 | 3071 | 3184 | 3071 | 539 | 5 |
| real | mp + cap | 90 | 333.7 ± 2.0 | 119 | 124 | 119 | 147 | 5 |
| peak | mp + cap | 90 | 410.5 ± 1.1 | 119 | 122 | 119 | 300 | 5 |
| real | mp + cap | 120 | 337.8 ± 2.3 | 147 | 152 | 147 | 172 | 5 |
| peak | mp + cap | 120 | 413.5 ± 1.4 | 146 | 147 | 146 | 321 | 5 |
| real | mp + deadline shield | 75 | 330.5 ± 2.0 | 75 | 75 | 76 | 126 | 5 |
| peak | mp + deadline shield | 75 | 410.2 ± 1.5 | 75 | 75 | 76 | 279 | 5 |
| real | mp + deadline shield | 90 | 332.3 ± 2.2 | 90 | 90 | 91 | 136 | 5 |
| peak | mp + deadline shield | 90 | 412.4 ± 1.1 | 90 | 90 | 91 | 295 | 5 |
| real | mp + deadline shield | 120 | 335.9 ± 1.7 | 120 | 120 | 121 | 158 | 5 |
| peak | mp + deadline shield | 120 | 412.2 ± 1.1 | 120 | 120 | 121 | 309 | 5 |
| real | actuated | 30 | 322.1 ± 3.7 | 73 | 79 | 73 | 84 | 5 |
| peak | actuated | 30 | 437.6 ± 4.3 | 102 | 105 | 102 | 407 | 5 |
| real | actuated | 40 | 319.9 ± 3.0 | 83 | 90 | 83 | 76 | 5 |
| peak | actuated | 40 | 427.5 ± 3.7 | 122 | 133 | 122 | 369 | 5 |
| real | actuated | 60 | 319.0 ± 2.2 | 100 | 113 | 100 | 73 | 5 |
| peak | actuated | 60 | 414.9 ± 1.2 | 169 | 176 | 169 | 313 | 5 |
