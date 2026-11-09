| Scenario | Controller | Avg travel time (s) | Avg queue (veh) | Throughput (veh) | Completed | Episodes |
|---|---|---|---|---|---|---|
| low | legacy_1s_cycle | 49.3 ± 0.3 | 0.03 ± 0.00 | 1067 ± 19 | 98.7% | 5 |
| low | fixed_time | 93.7 ± 2.4 | 6.44 ± 0.29 | 1062 ± 20 | 98.2% | 5 |
| low | max_pressure | 61.5 ± 0.3 | 1.64 ± 0.03 | 1065 ± 19 | 98.4% | 5 |
| low | actuated | 64.5 ± 0.9 | 1.90 ± 0.10 | 1064 ± 19 | 98.3% | 5 |
| low | ppo | 60.2 ± 0.6 | 1.43 ± 0.05 | 1066 ± 19 | 98.5% | 5 |
| low | ppo_pressure | 60.4 ± 0.5 | 1.49 ± 0.07 | 1064 ± 18 | 98.4% | 5 |
| low | dqn | 60.6 ± 0.7 | 1.46 ± 0.05 | 1065 ± 18 | 98.4% | 5 |
| medium | legacy_1s_cycle | 52.0 ± 0.4 | 0.07 ± 0.01 | 2160 ± 27 | 98.6% | 5 |
| medium | fixed_time | 119.6 ± 9.0 | 20.31 ± 2.49 | 2120 ± 18 | 96.8% | 5 |
| medium | max_pressure | 70.2 ± 0.9 | 5.51 ± 0.24 | 2146 ± 22 | 97.9% | 5 |
| medium | actuated | 77.7 ± 0.9 | 7.78 ± 0.26 | 2144 ± 25 | 97.9% | 5 |
| medium | ppo | 67.5 ± 0.5 | 4.78 ± 0.10 | 2151 ± 26 | 98.2% | 5 |
| medium | ppo_pressure | 67.7 ± 0.8 | 4.90 ± 0.19 | 2148 ± 25 | 98.1% | 5 |
| medium | dqn | 68.1 ± 0.4 | 4.85 ± 0.12 | 2148 ± 25 | 98.1% | 5 |
| high | legacy_1s_cycle | 59.1 ± 0.8 | 0.58 ± 0.08 | 3231 ± 45 | 98.3% | 5 |
| high | fixed_time | 314.5 ± 15.3 | 70.83 ± 3.06 | 2784 ± 39 | 92.6% | 5 |
| high | max_pressure | 112.5 ± 6.1 | 25.64 ± 2.72 | 3160 ± 44 | 96.1% | 5 |
| high | actuated | 96.0 ± 3.9 | 19.79 ± 1.81 | 3191 ± 43 | 97.1% | 5 |
| high | ppo | 81.3 ± 1.5 | 12.93 ± 0.65 | 3204 ± 44 | 97.5% | 5 |
| high | ppo_pressure | 81.5 ± 1.1 | 13.14 ± 0.46 | 3204 ± 44 | 97.5% | 5 |
| high | dqn | 86.3 ± 2.6 | 14.97 ± 1.18 | 3194 ± 43 | 97.2% | 5 |
| original | legacy_1s_cycle | 1344.6 ± 0.0 | 59.97 ± 0.00 | 2754 ± 0 | 91.9% | 1 |
| original | fixed_time | 1426.3 ± 0.0 | 83.32 ± 0.00 | 2300 ± 0 | 92.4% | 1 |
| original | max_pressure | 1072.3 ± 0.0 | 48.48 ± 0.00 | 4444 ± 0 | 96.1% | 1 |
| original | actuated | 1026.7 ± 0.0 | 51.27 ± 0.00 | 4722 ± 0 | 96.4% | 1 |
| original | ppo | 938.3 ± 0.0 | 43.54 ± 0.00 | 5235 ± 0 | 97.1% | 1 |
| original | ppo_pressure | 1263.5 ± 0.0 | 72.67 ± 0.00 | 3127 ± 0 | 93.9% | 1 |
| original | dqn | 1018.0 ± 0.0 | 35.34 ± 0.00 | 4748 ± 0 | 97.5% | 1 |

Worst-case waits (per vehicle, mean ± std over seeds of each episode's value):

| Scenario | Controller | 95th pct. time stopped (s) | Max time stopped (s) | Longest single stop (s) | 95th pct. trip (s) | Longest trip (s) |
|---|---|---|---|---|---|---|
| low | legacy_1s_cycle | 1 ± 0 | 7 ± 1 | 5 ± 1 | 66 ± 0 | 78 ± 3 |
| low | fixed_time | 102 ± 1 | 153 ± 15 | 107 ± 0 | 164 ± 2 | 223 ± 15 |
| low | max_pressure | 45 ± 1 | 263 ± 29 | 262 ± 28 | 107 ± 3 | 318 ± 39 |
| low | actuated | 41 ± 1 | 73 ± 6 | 54 ± 4 | 106 ± 2 | 144 ± 7 |
| low | ppo | 38 ± 3 | 133 ± 30 | 133 ± 30 | 99 ± 2 | 184 ± 27 |
| low | ppo_pressure | 40 ± 3 | 160 ± 46 | 154 ± 53 | 99 ± 3 | 214 ± 34 |
| low | dqn | 38 ± 2 | 133 ± 39 | 123 ± 22 | 100 ± 2 | 186 ± 47 |
| medium | legacy_1s_cycle | 3 ± 1 | 15 ± 3 | 12 ± 3 | 74 ± 1 | 94 ± 5 |
| medium | fixed_time | 196 ± 38 | 337 ± 76 | 116 ± 8 | 270 ± 39 | 420 ± 82 |
| medium | max_pressure | 71 ± 5 | 393 ± 79 | 393 ± 80 | 129 ± 3 | 448 ± 73 |
| medium | actuated | 69 ± 3 | 122 ± 12 | 87 ± 6 | 135 ± 3 | 193 ± 12 |
| medium | ppo | 55 ± 1 | 271 ± 50 | 267 ± 54 | 114 ± 2 | 303 ± 19 |
| medium | ppo_pressure | 58 ± 2 | 310 ± 63 | 308 ± 63 | 117 ± 3 | 350 ± 66 |
| medium | dqn | 53 ± 1 | 282 ± 95 | 276 ± 85 | 116 ± 1 | 336 ± 100 |
| high | legacy_1s_cycle | 14 ± 1 | 48 ± 9 | 35 ± 14 | 98 ± 3 | 147 ± 15 |
| high | fixed_time | 655 ± 34 | 927 ± 72 | 206 ± 69 | 773 ± 44 | 1026 ± 74 |
| high | max_pressure | 250 ± 24 | 674 ± 52 | 526 ± 102 | 312 ± 20 | 739 ± 53 |
| high | actuated | 116 ± 17 | 186 ± 21 | 111 ± 4 | 187 ± 17 | 264 ± 23 |
| high | ppo | 100 ± 11 | 420 ± 47 | 365 ± 61 | 159 ± 11 | 472 ± 44 |
| high | ppo_pressure | 108 ± 7 | 389 ± 46 | 366 ± 67 | 166 ± 7 | 439 ± 43 |
| high | dqn | 104 ± 10 | 627 ± 116 | 484 ± 113 | 164 ± 8 | 643 ± 130 |
| original | legacy_1s_cycle | 295 ± 0 | 353 ± 0 | 29 ± 0 | 500 ± 0 | 553 ± 0 |
| original | fixed_time | 312 ± 0 | 316 ± 0 | 106 ± 0 | 387 ± 0 | 392 ± 0 |
| original | max_pressure | 200 ± 0 | 228 ± 0 | 29 ± 0 | 298 ± 0 | 329 ± 0 |
| original | actuated | 119 ± 0 | 121 ± 0 | 47 ± 0 | 192 ± 0 | 196 ± 0 |
| original | ppo | 191 ± 0 | 294 ± 0 | 72 ± 0 | 271 ± 0 | 381 ± 0 |
| original | ppo_pressure | 671 ± 0 | 3254 ± 0 | 3235 ± 0 | 766 ± 0 | 819 ± 0 |
| original | dqn | 0 ± 0 | 3577 ± 0 | 3577 ± 0 | 40 ± 0 | 40 ± 0 |
