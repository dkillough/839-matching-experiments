# USAGE: python main.py <N>
# N = number of participants/pairings

import sys
import random
from gale_shapley import generate_random_preferences, stableMarriage

DEFAULT_N = 4

# TODO: Other algorithms.

def runGaleShapley(N):
    print(f"Run Gale-Shapley Algorithm with {N}x{N} pairings:")
    if N <= 0:
        print("Invalid number of participants; using defaults")
        N = DEFAULT_N
    prefer = generate_random_preferences(N, N)

    N = len(prefer[0])
    wPartner = stableMarriage(prefer)
    print("Woman", "Man")
    for i in range(N):
        print(i + N, "\t", wPartner[i])

if __name__ == "__main__":
    N = DEFAULT_N
    if len(sys.argv) != 2:
        print("Insufficient arguments; using defaults")
    else: 
        try:
            N = int(sys.argv[1])
            # M = int(sys.argv[2])
        except ValueError:
            print("Invalid arguments; using defaults")

    runGaleShapley(N)