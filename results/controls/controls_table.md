Flip rate by attack arm (flows correct before the attack; has_payload flows; models: base, seed0, seed1, seed2). Cells: mean over models ± standard deviation (flips/flows summed over models).

**DDoS-UDP_Flood**

| Attack arm | ε = 1 | ε = 4 | ε = 16 | ε = 32 |
|---|---|---|---|---|
| payload only (the constrained attack) | 0.0% ± 0.0 (0/376) | 0.5% ± 0.6 (2/376) | 1.8% ± 1.3 (7/376) | 2.9% ± 1.5 (11/376) |
| all bytes of channel 0 | 0.8% ± 1.0 (3/376) | 1.3% ± 1.3 (5/376) | 9.5% ± 5.5 (36/376) | 26.3% ± 6.9 (99/376) |
| all pixels (unconstrained) | 1.1% ± 1.5 (4/376) | 4.8% ± 2.1 (18/376) | 66.6% ± 22.6 (251/376) | 100.0% ± 0.0 (376/376) |
| random noise, payload only | 0.0% ± 0.0 (0/376) | 0.0% ± 0.0 (0/376) | 0.0% ± 0.0 (0/376) | 0.5% ± 0.6 (2/376) |
| random noise, all pixels | 0.0% ± 0.0 (0/376) | 0.3% ± 0.5 (1/376) | 1.1% ± 0.9 (4/376) | 1.9% ± 1.0 (7/376) |

**Recon-PortScan**

| Attack arm | ε = 1 | ε = 4 | ε = 16 | ε = 32 |
|---|---|---|---|---|
| payload only (the constrained attack) | 0.9% ± 1.1 (3/344) | 1.4% ± 1.5 (5/344) | 7.0% ± 1.9 (24/344) | 13.6% ± 3.1 (47/344) |
| all bytes of channel 0 | 1.4% ± 1.5 (5/344) | 7.0% ± 2.5 (24/344) | 22.4% ± 5.8 (77/344) | 61.8% ± 10.4 (213/344) |
| all pixels (unconstrained) | 5.8% ± 2.8 (20/344) | 24.2% ± 6.3 (83/344) | 92.6% ± 7.6 (319/344) | 99.7% ± 0.6 (343/344) |
| random noise, payload only | 0.0% ± 0.0 (0/344) | 0.0% ± 0.0 (0/344) | 0.6% ± 0.7 (2/344) | 0.9% ± 1.8 (3/344) |
| random noise, all pixels | 0.3% ± 0.6 (1/344) | 0.6% ± 0.7 (2/344) | 1.4% ± 1.5 (5/344) | 2.0% ± 2.4 (7/344) |
