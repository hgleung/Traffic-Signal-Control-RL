| Scenario | Controller | Avg travel time (s) | Avg queue (veh) | Throughput (veh) | Completed | Episodes |
|---|---|---|---|---|---|---|
| medium | lag | 72.3 ± 1.8 | 5.95 ± 0.50 | 2143 ± 22 | 97.8% | 5 |
| medium | lag_ds75 | 72.3 ± 1.8 | 5.95 ± 0.50 | 2143 ± 22 | 97.8% | 5 |
| high | lag | 220.0 ± 22.2 | 51.57 ± 4.64 | 2900 ± 13 | 92.9% | 5 |
| high | lag_ds75 | 223.4 ± 23.5 | 51.66 ± 4.21 | 2892 ± 12 | 92.7% | 5 |

Worst-case waits (per vehicle, mean ± std over seeds of each episode's value):

| Scenario | Controller | 95th pct. time stopped (s) | Max time stopped (s) | Longest single stop (s) | Longest red with a car waiting (s) | 95th pct. trip (s) | Longest trip (s) |
|---|---|---|---|---|---|---|---|
| medium | lag | 56 ± 4 | 109 ± 24 | 75 ± 3 | 75 ± 3 | 121 ± 5 | 179 ± 13 |
| medium | lag_ds75 | 56 ± 4 | 109 ± 24 | 75 ± 3 | 75 ± 3 | 121 ± 5 | 179 ± 13 |
| high | lag | 589 ± 41 | 1198 ± 287 | 449 ± 64 | 135 ± 69 | 658 ± 51 | 1305 ± 262 |
| high | lag_ds75 | 602 ± 48 | 1036 ± 161 | 377 ± 44 | 89 ± 3 | 694 ± 58 | 1125 ± 162 |
