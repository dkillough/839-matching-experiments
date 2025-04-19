import random
import sys
import json

def generate_random_preferences(N):
    """
    Generate random preference lists for N participants.
    Each participant ranks all others randomly.
    """
    men_preferences = [random.sample(range(N, 2 * N), N) for _ in range(N)]
    women_preferences = [random.sample(range(N), N) for _ in range(N)]
    return men_preferences, women_preferences

def generate_multiple_preferences(M, N):
    """
    Generate M sets of random preference lists for NxN participants.
    """
    all_preferences = []
    for _ in range(M):
        men_preferences, women_preferences = generate_random_preferences(N)
        all_preferences.append({
            "men_preferences": men_preferences,
            "women_preferences": women_preferences
        })
    return all_preferences

def write_preferences_to_json(filename, all_preferences):
    """
    Write the generated preferences to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(all_preferences, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python gen_pref.py <M> <N> <output_file>")
        sys.exit(1)

    try:
        M = int(sys.argv[1])
        N = int(sys.argv[2])
        output_file = sys.argv[3]
    except ValueError:
        print("Invalid input. M and N must be integers.")
        sys.exit(1)

    if M <= 0 or N <= 0:
        print("M and N must be positive integers.")
        sys.exit(1)

    all_preferences = generate_multiple_preferences(M, N)
    write_preferences_to_json(output_file, all_preferences)
    print(f"Preferences written to {output_file}")