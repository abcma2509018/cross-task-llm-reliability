#!/usr/bin/env python3
# Deterministic summary; does not read full trajectories or run experiments.
T=5.6389
print({"bytes_per_sample":512*T,"mean_T":T,"effective_KiB":512*T/1024,"padded_T32_KiB":512*32/1024})
