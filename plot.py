import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
from dynamic_smp_game import DynamicMarriageGame, load_preference_lists

def analyze_preference_lists(n_range=range(1, 11), max_turns=1000):
    """
    Analyze preference lists of different sizes and collect utility statistics
    for stable matchings.
    
    Args:
        n_range: Range of n values to analyze
        max_turns: Maximum number of turns to simulate for each preference list
        
    Returns:
        Dictionary with statistics for each n value
    """
    results = {}
    
    for n in n_range:
        print(f"Analyzing n={n}...")
        json_file = f"preference_lists/lists-n{n}.json"
        
        if not os.path.exists(json_file):
            print(f"File {json_file} not found, skipping.")
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            n_lists = len(data)
            print(f"Found {n_lists} preference lists for n={n}")
            
            # Initialize statistics
            n_results = {
                'stable_count': 0,
                'all_utilities': [],
                'men_utilities': [],
                'women_utilities': [],
                'min_utilities': [],
                'max_utilities': [],
                'avg_utilities': [],
                'turns_required': []
            }
            
            # Process each preference list
            for i, pref_list in enumerate(data):
                preferences = {
                    "men_preferences": pref_list["men_preferences"],
                    "women_preferences": pref_list["women_preferences"]
                }
                
                # Create game instance
                game = DynamicMarriageGame(n, preferences)
                
                # Simulate dynamics
                result = game.simulate_gale_shapley_dynamics(max_turns=max_turns)
                
                # Check if a stable matching was reached
                if game.is_stable():
                    n_results['stable_count'] += 1
                    n_results['turns_required'].append(game.turn)
                    
                    # Collect utilities
                    player_utilities = []
                    men_utilities = []
                    women_utilities = []
                    
                    for player, partner in game.couples.items():
                        if partner is not None:
                            utility = game.player_utilities[player]
                            player_utilities.append(utility)
                            
                            if player < n:
                                men_utilities.append(utility)
                            else:
                                women_utilities.append(utility)
                    
                    if player_utilities:
                        n_results['all_utilities'].extend(player_utilities)
                        n_results['men_utilities'].extend(men_utilities)
                        n_results['women_utilities'].extend(women_utilities)
                        n_results['min_utilities'].append(min(player_utilities))
                        n_results['max_utilities'].append(max(player_utilities))
                        n_results['avg_utilities'].append(sum(player_utilities) / len(player_utilities))
                        
                    print(f"  List {i}: Stable matching reached in {game.turn} turns")
                else:
                    print(f"  List {i}: No stable matching reached in {max_turns} turns")
            
            # Store results for this n
            results[n] = n_results
            
        except Exception as e:
            print(f"Error processing n={n}: {e}")
    
    return results

def plot_utility_statistics(results):
    """
    Create plots for utility statistics across different n values.
    
    Args:
        results: Results dictionary from analyze_preference_lists
    """
    # Extract data for plotting
    n_values = sorted(results.keys())
    
    # Calculate average statistics for each n
    avg_min_utility = []
    avg_max_utility = []
    avg_avg_utility = []
    avg_men_utility = []
    avg_women_utility = []
    stable_percentages = []
    avg_turns = []
    
    for n in n_values:
        n_results = results[n]
        
        if n_results['stable_count'] > 0:
            # Average of minimums, maximums, and averages
            avg_min_utility.append(sum(n_results['min_utilities']) / len(n_results['min_utilities']))
            avg_max_utility.append(sum(n_results['max_utilities']) / len(n_results['max_utilities']))
            avg_avg_utility.append(sum(n_results['avg_utilities']) / len(n_results['avg_utilities']))
            
            # Average men's and women's utilities
            if n_results['men_utilities']:
                avg_men_utility.append(sum(n_results['men_utilities']) / len(n_results['men_utilities']))
            else:
                avg_men_utility.append(0)
                
            if n_results['women_utilities']:
                avg_women_utility.append(sum(n_results['women_utilities']) / len(n_results['women_utilities']))
            else:
                avg_women_utility.append(0)
            
            # Average turns to reach stable matching
            avg_turns.append(sum(n_results['turns_required']) / len(n_results['turns_required']))
        else:
            # No stable matchings found
            avg_min_utility.append(0)
            avg_max_utility.append(0)
            avg_avg_utility.append(0)
            avg_men_utility.append(0)
            avg_women_utility.append(0)
            avg_turns.append(max_turns)
        
        # Percentage of stable matchings
        with open(f"preference_lists/lists-n{n}.json", 'r') as f:
            total_lists = len(json.load(f))
        stable_percentages.append(n_results['stable_count'] / total_lists * 100)
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Min, Max, and Average Utilities
    plt.subplot(2, 2, 1)
    plt.plot(n_values, avg_min_utility, 'r-', label='Min Utility')
    plt.plot(n_values, avg_max_utility, 'g-', label='Max Utility')
    plt.plot(n_values, avg_avg_utility, 'b-', label='Avg Utility')
    plt.xlabel('N (Number of men/women)')
    plt.ylabel('Average Utility')
    plt.title('Average Min, Max, and Avg Utilities by N')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Men vs. Women Utilities
    plt.subplot(2, 2, 2)
    plt.plot(n_values, avg_men_utility, 'b-', label='Men')
    plt.plot(n_values, avg_women_utility, 'r-', label='Women')
    plt.xlabel('N (Number of men/women)')
    plt.ylabel('Average Utility')
    plt.title('Average Utilities by Gender')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Percentage of Stable Matchings
    plt.subplot(2, 2, 3)
    plt.bar(n_values, stable_percentages)
    plt.xlabel('N (Number of men/women)')
    plt.ylabel('Percentage')
    plt.title('Percentage of Preference Lists with Stable Matchings')
    plt.ylim(0, 105)
    plt.grid(True)
    
    # Plot 4: Average Turns to Stable Matching
    plt.subplot(2, 2, 4)
    plt.plot(n_values, avg_turns, 'k-o')
    plt.xlabel('N (Number of men/women)')
    plt.ylabel('Average Turns')
    plt.title('Average Turns to Reach Stable Matching')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('utility_statistics.png')
    print("Plots saved as 'utility_statistics.png'")
    
    # Additional plot: Distribution of utilities for a specific n
    if 5 in results:
        plt.figure(figsize=(10, 6))
        plt.hist(results[5]['all_utilities'], bins=20, alpha=0.7, label='All')
        plt.hist(results[5]['men_utilities'], bins=20, alpha=0.7, label='Men')
        plt.hist(results[5]['women_utilities'], bins=20, alpha=0.7, label='Women')
        plt.xlabel('Utility')
        plt.ylabel('Frequency')
        plt.title('Distribution of Utilities for N=5')
        plt.legend()
        plt.grid(True)
        plt.savefig('utility_distribution_n5.png')
        print("Distribution plot saved as 'utility_distribution_n5.png'")

def main():
    # Get command line arguments
    max_turns = 1000
    if len(sys.argv) > 1:
        try:
            max_turns = int(sys.argv[1])
        except ValueError:
            print(f"Invalid max_turns value: {sys.argv[1]}")
            print(f"Using default: {max_turns}")
    
    # Analyze preference lists
    results = analyze_preference_lists(max_turns=max_turns)
    
    # Plot results
    plot_utility_statistics(results)
    
    # Print summary
    print("\nSummary:")
    for n, n_results in sorted(results.items()):
        print(f"N={n}: {n_results['stable_count']} stable matchings")
        if n_results['stable_count'] > 0:
            print(f"  Avg Min Utility: {sum(n_results['min_utilities']) / len(n_results['min_utilities']):.4f}")
            print(f"  Avg Max Utility: {sum(n_results['max_utilities']) / len(n_results['max_utilities']):.4f}")
            print(f"  Avg Avg Utility: {sum(n_results['avg_utilities']) / len(n_results['avg_utilities']):.4f}")
            print(f"  Avg Turns: {sum(n_results['turns_required']) / len(n_results['turns_required']):.2f}")

if __name__ == "__main__":
    main()