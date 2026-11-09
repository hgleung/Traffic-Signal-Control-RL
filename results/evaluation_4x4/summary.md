| Scenario | Controller | Avg travel time (s) | Avg queue (veh) | Throughput (veh) | Completed | Episodes |
|---|---|---|---|---|---|---|
| low | legacy_1s_cycle | 104.0 ± 0.8 | 0.02 ± 0.00 | 2138 ± 32 | 98.2% | 5 |
| low | fixed_time | 244.3 ± 2.9 | 5.13 ± 0.13 | 2084 ± 31 | 95.7% | 5 |
| low | max_pressure | 133.1 ± 1.8 | 0.97 ± 0.04 | 2130 ± 34 | 97.8% | 5 |
| low | actuated | 147.2 ± 2.7 | 1.31 ± 0.08 | 2124 ± 32 | 97.5% | 5 |
| low | ppo | 129.3 ± 1.6 | 0.83 ± 0.03 | 2131 ± 33 | 97.9% | 5 |
| low | ppo_neighbors | 130.0 ± 1.6 | 0.84 ± 0.04 | 2131 ± 32 | 97.9% | 5 |
| low | ppo_1x2_transfer | 133.3 ± 1.4 | 0.94 ± 0.03 | 2128 ± 33 | 97.7% | 5 |
| medium | legacy_1s_cycle | 107.8 ± 0.9 | 0.05 ± 0.00 | 4238 ± 39 | 98.2% | 5 |
| medium | fixed_time | 311.4 ± 8.0 | 14.44 ± 0.46 | 4049 ± 37 | 93.9% | 5 |
| medium | max_pressure | 154.5 ± 1.9 | 3.26 ± 0.10 | 4209 ± 39 | 97.5% | 5 |
| medium | actuated | 175.6 ± 2.7 | 4.65 ± 0.18 | 4199 ± 41 | 97.3% | 5 |
| medium | ppo | 150.2 ± 2.0 | 2.94 ± 0.10 | 4216 ± 36 | 97.6% | 5 |
| medium | ppo_neighbors | 150.5 ± 2.5 | 2.94 ± 0.14 | 4214 ± 42 | 97.6% | 5 |
| medium | ppo_1x2_transfer | 153.0 ± 2.0 | 3.13 ± 0.11 | 4215 ± 40 | 97.6% | 5 |
| high | legacy_1s_cycle | 114.0 ± 0.4 | 0.11 ± 0.00 | 6330 ± 88 | 98.0% | 5 |
| high | fixed_time | 439.4 ± 8.4 | 24.46 ± 0.25 | 5513 ± 67 | 91.9% | 5 |
| high | max_pressure | 216.1 ± 9.5 | 10.97 ± 1.06 | 6235 ± 74 | 96.5% | 5 |
| high | actuated | 231.7 ± 7.7 | 12.23 ± 0.57 | 6192 ± 66 | 95.9% | 5 |
| high | ppo | 190.3 ± 4.3 | 8.26 ± 0.48 | 6273 ± 86 | 97.1% | 5 |
| high | ppo_neighbors | 195.3 ± 5.9 | 8.73 ± 0.60 | 6273 ± 87 | 97.1% | 5 |
| high | ppo_1x2_transfer | 193.6 ± 3.7 | 8.71 ± 0.41 | 6271 ± 87 | 97.1% | 5 |

Worst-case waits (per vehicle, mean ± std over seeds of each episode's value):

| Scenario | Controller | 95th pct. time stopped (s) | Max time stopped (s) | Longest single stop (s) | 95th pct. trip (s) | Longest trip (s) |
|---|---|---|---|---|---|---|
| low | legacy_1s_cycle | 3 ± 0 | 12 ± 4 | 7 ± 4 | 173 ± 4 | 306 ± 12 |
| low | fixed_time | 278 ± 5 | 583 ± 59 | 109 ± 1 | 447 ± 7 | 877 ± 81 |
| low | max_pressure | 90 ± 5 | 313 ± 51 | 243 ± 19 | 252 ± 10 | 575 ± 59 |
| low | actuated | 89 ± 5 | 190 ± 49 | 61 ± 3 | 264 ± 11 | 499 ± 68 |
| low | ppo | 73 ± 1 | 215 ± 33 | 170 ± 20 | 238 ± 5 | 443 ± 57 |
| low | ppo_neighbors | 72 ± 2 | 219 ± 24 | 153 ± 21 | 237 ± 2 | 474 ± 43 |
| low | ppo_1x2_transfer | 77 ± 2 | 260 ± 22 | 201 ± 20 | 243 ± 6 | 502 ± 46 |
| medium | legacy_1s_cycle | 4 ± 0 | 16 ± 3 | 10 ± 2 | 181 ± 3 | 333 ± 9 |
| medium | fixed_time | 469 ± 23 | 817 ± 47 | 115 ± 1 | 636 ± 23 | 1112 ± 77 |
| medium | max_pressure | 142 ± 6 | 539 ± 103 | 401 ± 68 | 307 ± 5 | 807 ± 116 |
| medium | actuated | 148 ± 3 | 358 ± 39 | 96 ± 4 | 322 ± 3 | 657 ± 31 |
| medium | ppo | 117 ± 4 | 426 ± 52 | 324 ± 57 | 287 ± 5 | 681 ± 50 |
| medium | ppo_neighbors | 116 ± 4 | 381 ± 15 | 307 ± 49 | 286 ± 5 | 631 ± 72 |
| medium | ppo_1x2_transfer | 129 ± 5 | 471 ± 70 | 383 ± 68 | 298 ± 7 | 722 ± 70 |
| high | legacy_1s_cycle | 7 ± 0 | 38 ± 4 | 26 ± 5 | 192 ± 3 | 371 ± 10 |
| high | fixed_time | 559 ± 14 | 1159 ± 110 | 117 ± 2 | 743 ± 31 | 1498 ± 148 |
| high | max_pressure | 340 ± 40 | 1304 ± 215 | 632 ± 76 | 519 ± 38 | 1631 ± 247 |
| high | actuated | 272 ± 12 | 616 ± 70 | 117 ± 4 | 443 ± 12 | 957 ± 90 |
| high | ppo | 222 ± 16 | 734 ± 118 | 413 ± 39 | 400 ± 16 | 1044 ± 152 |
| high | ppo_neighbors | 232 ± 19 | 787 ± 138 | 488 ± 74 | 414 ± 20 | 1104 ± 131 |
| high | ppo_1x2_transfer | 243 ± 15 | 984 ± 193 | 524 ± 90 | 419 ± 13 | 1270 ± 246 |
