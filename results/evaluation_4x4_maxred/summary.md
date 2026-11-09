| Scenario | Controller | Avg travel time (s) | Avg queue (veh) | Throughput (veh) | Completed | Episodes |
|---|---|---|---|---|---|---|
| low | actuated | 147.2 ± 2.7 | 1.31 ± 0.08 | 2124 ± 32 | 97.5% | 5 |
| low | max_pressure | 133.1 ± 1.8 | 0.97 ± 0.04 | 2130 ± 34 | 97.8% | 5 |
| low | ppo | 129.3 ± 1.6 | 0.83 ± 0.03 | 2131 ± 33 | 97.9% | 5 |
| low | ppo_maxred60 | 129.9 ± 1.6 | 0.84 ± 0.03 | 2131 ± 33 | 97.8% | 5 |
| low | ppo_maxred90 | 130.2 ± 1.5 | 0.85 ± 0.03 | 2130 ± 33 | 97.8% | 5 |
| low | ppo_maxred120 | 129.6 ± 1.7 | 0.84 ± 0.04 | 2131 ± 35 | 97.9% | 5 |
| low | ppo_retrained_maxred60 | 129.5 ± 2.5 | 0.83 ± 0.06 | 2130 ± 32 | 97.8% | 5 |
| medium | actuated | 175.6 ± 2.7 | 4.65 ± 0.18 | 4199 ± 41 | 97.3% | 5 |
| medium | max_pressure | 154.5 ± 1.9 | 3.26 ± 0.10 | 4209 ± 39 | 97.5% | 5 |
| medium | ppo | 150.2 ± 2.0 | 2.94 ± 0.10 | 4216 ± 36 | 97.6% | 5 |
| medium | ppo_maxred60 | 154.2 ± 2.6 | 3.18 ± 0.14 | 4215 ± 41 | 97.6% | 5 |
| medium | ppo_maxred90 | 151.3 ± 2.6 | 3.01 ± 0.13 | 4214 ± 42 | 97.6% | 5 |
| medium | ppo_maxred120 | 150.6 ± 2.5 | 2.96 ± 0.14 | 4214 ± 41 | 97.6% | 5 |
| medium | ppo_retrained_maxred60 | 153.8 ± 2.3 | 3.14 ± 0.12 | 4214 ± 42 | 97.6% | 5 |
| high | actuated | 231.7 ± 7.7 | 12.23 ± 0.57 | 6192 ± 66 | 95.9% | 5 |
| high | max_pressure | 216.1 ± 9.5 | 10.97 ± 1.06 | 6235 ± 74 | 96.5% | 5 |
| high | ppo | 190.3 ± 4.3 | 8.26 ± 0.48 | 6273 ± 86 | 97.1% | 5 |
| high | ppo_maxred60 | 237.3 ± 7.5 | 12.09 ± 0.78 | 6158 ± 49 | 95.4% | 5 |
| high | ppo_maxred90 | 208.4 ± 6.0 | 9.89 ± 0.59 | 6256 ± 81 | 96.8% | 5 |
| high | ppo_maxred120 | 199.9 ± 7.2 | 9.12 ± 0.74 | 6267 ± 77 | 97.0% | 5 |
| high | ppo_retrained_maxred60 | 235.3 ± 8.5 | 11.75 ± 0.84 | 6156 ± 53 | 95.4% | 5 |

Worst-case waits (per vehicle, mean ± std over seeds of each episode's value):

| Scenario | Controller | 95th pct. time stopped (s) | Max time stopped (s) | Longest single stop (s) | 95th pct. trip (s) | Longest trip (s) |
|---|---|---|---|---|---|---|
| low | actuated | 89 ± 5 | 190 ± 49 | 61 ± 3 | 264 ± 11 | 499 ± 68 |
| low | max_pressure | 90 ± 5 | 313 ± 51 | 243 ± 19 | 252 ± 10 | 575 ± 59 |
| low | ppo | 73 ± 1 | 215 ± 33 | 170 ± 20 | 238 ± 5 | 443 ± 57 |
| low | ppo_maxred60 | 72 ± 2 | 183 ± 7 | 82 ± 6 | 236 ± 6 | 478 ± 18 |
| low | ppo_maxred90 | 73 ± 3 | 206 ± 30 | 106 ± 1 | 239 ± 7 | 439 ± 49 |
| low | ppo_maxred120 | 73 ± 2 | 215 ± 42 | 133 ± 5 | 239 ± 9 | 490 ± 60 |
| low | ppo_retrained_maxred60 | 67 ± 4 | 181 ± 52 | 76 ± 2 | 234 ± 11 | 473 ± 89 |
| medium | actuated | 148 ± 3 | 358 ± 39 | 96 ± 4 | 322 ± 3 | 657 ± 31 |
| medium | max_pressure | 142 ± 6 | 539 ± 103 | 401 ± 68 | 307 ± 5 | 807 ± 116 |
| medium | ppo | 117 ± 4 | 426 ± 52 | 324 ± 57 | 287 ± 5 | 681 ± 50 |
| medium | ppo_maxred60 | 118 ± 6 | 328 ± 34 | 90 ± 3 | 290 ± 6 | 640 ± 29 |
| medium | ppo_maxred90 | 117 ± 4 | 333 ± 51 | 111 ± 4 | 285 ± 6 | 620 ± 77 |
| medium | ppo_maxred120 | 121 ± 4 | 326 ± 39 | 141 ± 5 | 287 ± 4 | 625 ± 50 |
| medium | ppo_retrained_maxred60 | 115 ± 3 | 295 ± 25 | 84 ± 6 | 287 ± 5 | 581 ± 59 |
| high | actuated | 272 ± 12 | 616 ± 70 | 117 ± 4 | 443 ± 12 | 957 ± 90 |
| high | max_pressure | 340 ± 40 | 1304 ± 215 | 632 ± 76 | 519 ± 38 | 1631 ± 247 |
| high | ppo | 222 ± 16 | 734 ± 118 | 413 ± 39 | 400 ± 16 | 1044 ± 152 |
| high | ppo_maxred60 | 302 ± 26 | 806 ± 126 | 127 ± 37 | 485 ± 28 | 1183 ± 176 |
| high | ppo_maxred90 | 241 ± 15 | 692 ± 150 | 120 ± 2 | 424 ± 19 | 1057 ± 186 |
| high | ppo_maxred120 | 229 ± 20 | 658 ± 103 | 152 ± 14 | 414 ± 24 | 1006 ± 134 |
| high | ppo_retrained_maxred60 | 295 ± 23 | 794 ± 135 | 148 ± 57 | 480 ± 29 | 1141 ± 183 |
