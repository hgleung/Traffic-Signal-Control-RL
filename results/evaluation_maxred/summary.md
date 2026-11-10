| Scenario | Controller | Avg travel time (s) | Avg queue (veh) | Throughput (veh) | Completed | Episodes |
|---|---|---|---|---|---|---|
| low | actuated | 64.5 ± 0.9 | 1.90 ± 0.10 | 1064 ± 19 | 98.3% | 5 |
| low | max_pressure | 61.5 ± 0.3 | 1.64 ± 0.03 | 1065 ± 19 | 98.4% | 5 |
| low | ppo | 60.2 ± 0.6 | 1.43 ± 0.05 | 1066 ± 19 | 98.5% | 5 |
| low | ppo_maxred60 | 60.5 ± 0.4 | 1.45 ± 0.04 | 1066 ± 19 | 98.5% | 5 |
| low | ppo_maxred90 | 60.3 ± 0.5 | 1.44 ± 0.03 | 1066 ± 19 | 98.5% | 5 |
| low | ppo_maxred120 | 60.2 ± 0.6 | 1.42 ± 0.05 | 1066 ± 19 | 98.5% | 5 |
| low | ppo_retrained_maxred60 | 64.2 ± 0.9 | 2.03 ± 0.10 | 1065 ± 20 | 98.4% | 5 |
| medium | actuated | 77.7 ± 0.9 | 7.78 ± 0.26 | 2144 ± 25 | 97.9% | 5 |
| medium | max_pressure | 70.2 ± 0.9 | 5.51 ± 0.24 | 2146 ± 22 | 97.9% | 5 |
| medium | ppo | 67.5 ± 0.5 | 4.78 ± 0.10 | 2151 ± 26 | 98.2% | 5 |
| medium | ppo_maxred60 | 69.2 ± 0.9 | 5.18 ± 0.23 | 2147 ± 24 | 98.0% | 5 |
| medium | ppo_maxred90 | 68.1 ± 0.6 | 4.92 ± 0.11 | 2149 ± 26 | 98.1% | 5 |
| medium | ppo_maxred120 | 68.2 ± 0.8 | 4.96 ± 0.21 | 2148 ± 25 | 98.1% | 5 |
| medium | ppo_retrained_maxred60 | 72.0 ± 0.8 | 5.98 ± 0.14 | 2146 ± 24 | 98.0% | 5 |
| high | actuated | 96.0 ± 3.9 | 19.79 ± 1.81 | 3191 ± 43 | 97.1% | 5 |
| high | max_pressure | 112.5 ± 6.1 | 25.64 ± 2.72 | 3160 ± 44 | 96.1% | 5 |
| high | ppo | 81.3 ± 1.5 | 12.93 ± 0.65 | 3204 ± 44 | 97.5% | 5 |
| high | ppo_maxred60 | 107.0 ± 9.8 | 23.19 ± 4.09 | 3164 ± 36 | 96.3% | 5 |
| high | ppo_maxred90 | 86.3 ± 1.6 | 14.88 ± 0.76 | 3206 ± 47 | 97.5% | 5 |
| high | ppo_maxred120 | 82.8 ± 1.1 | 13.51 ± 0.46 | 3206 ± 46 | 97.5% | 5 |
| high | ppo_retrained_maxred60 | 111.9 ± 12.5 | 25.36 ± 5.36 | 3142 ± 38 | 95.7% | 5 |

Worst-case waits (per vehicle, mean ± std over seeds of each episode's value):

| Scenario | Controller | 95th pct. time stopped (s) | Max time stopped (s) | Longest single stop (s) | 95th pct. trip (s) | Longest trip (s) |
|---|---|---|---|---|---|---|
| low | actuated | 41 ± 1 | 73 ± 6 | 54 ± 4 | 106 ± 2 | 144 ± 7 |
| low | max_pressure | 45 ± 1 | 263 ± 29 | 262 ± 28 | 107 ± 3 | 318 ± 39 |
| low | ppo | 38 ± 3 | 133 ± 30 | 133 ± 30 | 99 ± 2 | 184 ± 27 |
| low | ppo_maxred60 | 40 ± 2 | 80 ± 2 | 79 ± 3 | 100 ± 1 | 143 ± 7 |
| low | ppo_maxred90 | 38 ± 3 | 106 ± 4 | 105 ± 4 | 99 ± 2 | 165 ± 14 |
| low | ppo_maxred120 | 38 ± 3 | 120 ± 16 | 120 ± 16 | 99 ± 2 | 171 ± 16 |
| low | ppo_retrained_maxred60 | 59 ± 7 | 83 ± 4 | 79 ± 1 | 113 ± 3 | 153 ± 7 |
| medium | actuated | 69 ± 3 | 122 ± 12 | 87 ± 6 | 135 ± 3 | 193 ± 12 |
| medium | max_pressure | 71 ± 5 | 393 ± 79 | 393 ± 80 | 129 ± 3 | 448 ± 73 |
| medium | ppo | 55 ± 1 | 271 ± 50 | 267 ± 54 | 114 ± 2 | 303 ± 19 |
| medium | ppo_maxred60 | 58 ± 2 | 109 ± 11 | 82 ± 4 | 117 ± 1 | 178 ± 12 |
| medium | ppo_maxred90 | 56 ± 1 | 133 ± 16 | 111 ± 3 | 116 ± 1 | 204 ± 16 |
| medium | ppo_maxred120 | 57 ± 3 | 158 ± 22 | 137 ± 2 | 116 ± 1 | 228 ± 28 |
| medium | ppo_retrained_maxred60 | 66 ± 2 | 120 ± 10 | 82 ± 3 | 122 ± 2 | 186 ± 6 |
| high | actuated | 116 ± 17 | 186 ± 21 | 111 ± 4 | 187 ± 17 | 264 ± 23 |
| high | max_pressure | 250 ± 24 | 674 ± 52 | 526 ± 102 | 312 ± 20 | 739 ± 53 |
| high | ppo | 100 ± 11 | 420 ± 47 | 365 ± 61 | 159 ± 11 | 472 ± 44 |
| high | ppo_maxred60 | 152 ± 32 | 293 ± 56 | 149 ± 52 | 234 ± 38 | 381 ± 56 |
| high | ppo_maxred90 | 95 ± 3 | 201 ± 24 | 116 ± 6 | 157 ± 5 | 269 ± 25 |
| high | ppo_maxred120 | 95 ± 5 | 246 ± 31 | 141 ± 3 | 158 ± 4 | 303 ± 32 |
| high | ppo_retrained_maxred60 | 186 ± 61 | 388 ± 176 | 243 ± 122 | 266 ± 61 | 461 ± 155 |
