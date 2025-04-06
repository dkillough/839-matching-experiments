import random
import sys

# Naiive implementation of the Gale-Shapley algorithm from https://www.geeksforgeeks.org/stable-marriage-problem/
# Adapted for random preference lists and command line args
# Usage: python gale-shapley.py <N>

def wPrefersM1OverM(prefer, w, m, m1):
    N = len(prefer[0])
    for i in range(N):
        if prefer[w][i] == m1:
            return True
        if prefer[w][i] == m:
            return False
    return False


# Implements the stable marriage algorithm
def stableMarriage(prefer):
    N = len(prefer[0])
    wPartner = [-1] * N  # Stores women's partners
    mFree = [False] * N  # Tracks free men
    freeCount = N

    while freeCount > 0:
        m = next(i for i in range(N) if not mFree[i])
        for i in range(N):
            if mFree[m]:
                break
            w = prefer[m][i]
            if wPartner[w - N] == -1:
                wPartner[w - N] = m
                mFree[m] = True
                freeCount -= 1
            else:
                m1 = wPartner[w - N]
                if not wPrefersM1OverM(prefer, w, m, m1):
                    wPartner[w - N] = m
                    mFree[m] = True
                    mFree[m1] = False
    return wPartner

def generate_random_preferences(N, M):
    men_preferences = [random.sample(range(N, N + M), M) for _ in range(N)]
    women_preferences = [random.sample(range(N), N) for _ in range(M)]
    return men_preferences + women_preferences


if __name__ == "__main__":
    N = 4 
    if len(sys.argv) != 2:
        print("Insufficient arguments; using defaults")
    else: 
        try:
            N = int(sys.argv[1])
            # M = int(sys.argv[2])
        except ValueError:
            print("Invalid arguments; using defaults")
    
    prefer = generate_random_preferences(N, N)

    N = len(prefer[0])
    wPartner = stableMarriage(prefer)
    print("Woman", "Man")
    for i in range(N):
        print(i + N, "\t", wPartner[i])
