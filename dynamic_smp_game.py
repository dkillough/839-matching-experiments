import json
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Callable
import copy
import sys
import random

class DynamicMarriageGame:
    """Implementation of the dynamic marriage game with two-stage proposal/acceptance mechanism"""
    
    def __init__(self, n: int, preferences: Dict = None, utility_params: Dict = None):
        self.n = n
        self.total_players = 2 * n
        self.couples = {} 
        self.turn = 0
        self.history = []  
        self.player_utilities = {i: 0 for i in range(self.total_players)} 
        
        # Men are indexed 0 to n-1, women are indexed n to 2n-1
        self.men_indices = list(range(n))
        self.women_indices = list(range(n, self.total_players))

        # Initialize preference lists if provided
        if preferences:
            self.men_preferences = preferences.get('men_preferences', [])
            self.women_preferences = preferences.get('women_preferences', [])
            # Create utility functions from preferences
            self.initialize_utility_from_preferences()
        else:
            # Default utility functions
            self.men_preferences = []
            self.women_preferences = []
            self.initialize_default_utility()

        # Override utility parameters if provided
        if utility_params:
            for key, value in utility_params.items():
                setattr(self, key, value)

    def initialize_utility_from_preferences(self):
        """Initialize utility functions based on preference lists"""
        # f(u,v): utility player u receives when coupled with player v
        self.f = {}
        
        # Create f utility based on preferences for men
        for m in range(self.n):
            self.f[m] = {}
            for idx, w in enumerate(self.men_preferences[m]):
                # Higher utility for higher preferences
                self.f[m][w] = (self.n - idx) / self.n
            # Utility for being single (LOWER than being matched)
            self.f[m][0] = -0.1
            
        # Create f utility based on preferences for women
        for w_idx in range(self.n):
            w = w_idx + self.n  # Women are indexed from n to 2n-1
            self.f[w] = {}
            for idx, m in enumerate(self.women_preferences[w_idx]):
                self.f[w][m] = (self.n - idx) / self.n
            # Utility for being single (LOWER than being matched)
            self.f[w][0] = -0.1
        
        # p_→(u,v): utility from proposing to player v (non-positive)
        self.p_propose = {}
        # Initialize for men (0 to n-1)
        for m in range(self.n):
            self.p_propose[m] = {}
            # Men can propose to women (n to 2n-1)
            for w in range(self.n, self.total_players):
                self.p_propose[m][w] = -0.05  # Small cost for proposing
            # Men can't propose to men or themselves
            for m2 in range(self.n):
                self.p_propose[m][m2] = -0.2  # Higher penalty for invalid proposals
            self.p_propose[m][0] = 0  # No cost for doing nothing
        
        # Initialize for women (n to 2n-1)
        for w in range(self.n, self.total_players):
            self.p_propose[w] = {}
            # Women can propose to men (0 to n-1)
            for m in range(self.n):
                self.p_propose[w][m] = -0.05  # Small cost for proposing
            # Women can't propose to women or themselves
            for w2 in range(self.n, self.total_players):
                self.p_propose[w][w2] = -0.2  # Higher penalty for invalid proposals
            self.p_propose[w][0] = 0  # No cost for doing nothing
        
        # p_←(u,v): utility from receiving a proposal from player v
        self.p_receive = {}
        # Initialize for men (0 to n-1)
        for m in range(self.n):
            self.p_receive[m] = {}
            # Men can receive proposals from women (n to 2n-1)
            for w in range(self.n, self.total_players):
                # Get woman's index in man's preferences
                try:
                    w_rank = self.men_preferences[m].index(w)
                    self.p_receive[m][w] = 0.05 * (self.n - w_rank) / self.n
                except ValueError:
                    self.p_receive[m][w] = 0.01  # Minimal utility for receiving proposal from unlisted woman
            # Men don't receive proposals from men
            for m2 in range(self.n):
                self.p_receive[m][m2] = 0  # No utility for invalid proposals
        
        # Initialize for women (n to 2n-1)
        for w_idx in range(self.n):
            w = w_idx + self.n
            self.p_receive[w] = {}
            # Women can receive proposals from men (0 to n-1)
            for m in range(self.n):
                try:
                    m_rank = self.women_preferences[w_idx].index(m)
                    self.p_receive[w][m] = 0.05 * (self.n - m_rank) / self.n
                except ValueError:
                    self.p_receive[w][m] = 0.01  # Minimal utility for receiving proposal from unlisted man
            # Women don't receive proposals from women
            for w2 in range(self.n, self.total_players):
                self.p_receive[w][w2] = 0  # No utility for invalid proposals
        
        # r(u,v): utility from being rejected when proposing to player v (non-positive)
        self.r = {}
        for i in range(self.total_players):
            self.r[i] = {}
            for j in range(self.total_players):
                self.r[i][j] = -0.1 if i != j else 0  # Negative utility for being rejected
        
        # c(u,v): utility from successfully forming a couple when proposing to player v
        self.c = {}
        for i in range(self.total_players):
            self.c[i] = {}
            for j in range(self.total_players):
                self.c[i][j] = 0.2 if i != j else 0  # Positive utility for successfully forming a couple
        
        # b(u,v): utility when player v breaks a couple with player u (non-positive)
        self.b = {}
        for i in range(self.total_players):
            self.b[i] = {}
            for j in range(self.total_players):
                self.b[i][j] = -0.3 if i != j else 0  # Negative utility for being broken up with

    def initialize_default_utility(self):
        """Initialize default utility functions"""
        # Default utility values
        self.f = {i: {j: 0.5 if i != j else 0 for j in range(self.total_players)} for i in range(self.total_players)}
        for i in range(self.total_players):
            self.f[i][0] = -0.1  # Negative utility for being single
            
        self.p_propose = {i: {j: -0.05 if i != j else 0 for j in range(self.total_players)} for i in range(self.total_players)}
        self.p_receive = {i: {j: 0.05 if i != j else 0 for j in range(self.total_players)} for i in range(self.total_players)}
        self.r = {i: {j: -0.1 if i != j else 0 for j in range(self.total_players)} for i in range(self.total_players)}
        self.c = {i: {j: 0.2 if i != j else 0 for j in range(self.total_players)} for i in range(self.total_players)}
        self.b = {i: {j: -0.3 if i != j else 0 for j in range(self.total_players)} for i in range(self.total_players)}

    def play_turn(self, stage1_actions, stage2_actions):
        """
        Play one turn of the game with the given actions.
        
        Args:
            stage1_actions: Dictionary mapping players to their proposals (or None for no proposal)
            stage2_actions: Dictionary mapping players to their acceptances (or None for no acceptance)
            
        Returns:
            Updated game state and utilities earned in this turn
        """
        self.turn += 1
        turn_utilities = {i: 0 for i in range(self.total_players)}
        
        proposals_received = {i: [] for i in range(self.total_players)}
        proposals_made = {}
        
        # Process stage 1 actions (proposals)
        for player, proposal_to in stage1_actions.items():
            if proposal_to is not None:
                if proposal_to not in proposals_received:
                    continue  # Skip invalid proposals
                
                proposals_made[player] = proposal_to
                proposals_received[proposal_to].append(player)
                
                turn_utilities[player] += self.p_propose[player].get(proposal_to, 0)
                turn_utilities[proposal_to] += self.p_receive[proposal_to].get(player, 0)
        
        new_couples = copy.deepcopy(self.couples)  # !! Start with existing couples
        
        for player, accept in stage2_actions.items():
            if accept is not None:
                # Check if the accepted player proposed to this player or vice versa
                valid_acceptance = (accept in proposals_received[player]) or (proposals_made.get(player) == accept)
                
                if valid_acceptance:
                    # Check if the other player also accepted
                    other_accepted = stage2_actions.get(accept) == player
                    
                    if other_accepted:
                        # Both players accepted each other, form a new couple
                        new_couples[player] = accept
                        new_couples[accept] = player
                        
                        # Apply coupling utilities
                        if player in proposals_made and proposals_made[player] == accept:
                            turn_utilities[player] += self.c[player].get(accept, 0)
                                
                        if accept in proposals_made and proposals_made[accept] == player:
                            turn_utilities[accept] += self.c[accept].get(player, 0)
        
        # Process rejections
        for player, proposal_to in proposals_made.items():
            if player not in new_couples or new_couples[player] != proposal_to:
                # Player was rejected
                turn_utilities[player] += self.r[player].get(proposal_to, 0)
        
        # Process breakups
        for player, current_partner in self.couples.items():
            if player in new_couples:
                if new_couples[player] != current_partner:
                    # Player broke up with their current partner
                    turn_utilities[current_partner] += self.b[current_partner].get(player, 0)
            else:
                # Player is now single
                pass
        
        self.couples = new_couples
        
        # Apply utilities from being in a couple (!! passive gains per turn!!)
        for player, partner in self.couples.items():
            turn_utilities[player] += self.f[player].get(partner, 0)
        
        # Apply utilities from being single
        for player in range(self.total_players):
            if player not in self.couples:
                turn_utilities[player] += self.f[player].get(0, 0)
        
        # Update cumulative utilities
        for player, utility in turn_utilities.items():
            self.player_utilities[player] += utility
        
        # Record history
        self.history.append({
            'turn': self.turn,
            'stage1_actions': stage1_actions,
            'stage2_actions': stage2_actions,
            'couples': copy.deepcopy(self.couples),
            'turn_utilities': turn_utilities
        })
        
        return {
            'couples': self.couples,
            'turn_utilities': turn_utilities,
            'cumulative_utilities': self.player_utilities
        }
    
    def is_stable(self) -> bool:
        if not self.couples:
            return False  # No matching yet
        
        # Get all matched men and women
        matched_men = [m for m in range(self.n) if m in self.couples]
        
        # Check for blocking pairs
        for m in matched_men:
            w = self.couples[m]
            
            # Make sure w is a woman (in women_indices)
            if w not in self.women_indices:
                continue
                
            # Check all other women
            for w_prime in self.women_indices:
                # Skip current partner and single women
                if w_prime == w or w_prime not in self.couples:
                    continue
                    
                m_prime = self.couples[w_prime]
                
                # Make sure m_prime is a man
                if m_prime not in self.men_indices:
                    continue
                
                # Check if (m, w_prime) would prefer each other to current partners
                try:
                    if self.f[m].get(w_prime, 0) > self.f[m].get(w, 0) and \
                       self.f[w_prime].get(m, 0) > self.f[w_prime].get(m_prime, 0):
                        return False  # Blocking pair found
                except KeyError:
                    continue  # Skip if preferences not defined
                    
        return True

    def get_matching(self) -> Dict[int, int]:
        """
        Get the current matching as a dictionary.
        """
        return self.couples
    
    def get_stability_incentives(self) -> Dict[Tuple[int, int], float]:
        """
        Calculate incentives for deviation from current matching.
        """
        incentives = {}
        
        if not self.couples:
            return incentives  # No matching yet
            
        for m in range(self.n):
            if m not in self.couples:
                continue  # Skip single players
                
            w = self.couples[m]
            
            if w not in self.women_indices:
                continue
                
            for w_prime in self.women_indices:
                # Skip current partner and single women
                if w_prime == w or w_prime not in self.couples:
                    continue
                    
                m_prime = self.couples[w_prime]
                
                if m_prime not in self.men_indices:
                    continue
                
                # Calculate incentive as the sum of utility improvements
                try:
                    m_improvement = self.f[m].get(w_prime, 0) - self.f[m].get(w, 0)
                    w_prime_improvement = self.f[w_prime].get(m, 0) - self.f[w_prime].get(m_prime, 0)
                    
                    if m_improvement > 0 and w_prime_improvement > 0:
                        incentives[(m, w_prime)] = m_improvement + w_prime_improvement
                except KeyError:
                    continue  # Skip if preferences not defined
                    
        return incentives
    
    def get_nash_equilibrium_strategy(self):
        """
        Determine if the current state represents a Nash equilibrium strategy.
        """
        if not self.is_stable():
            return {
                'is_nash_equilibrium': False,
                'reason': 'Current matching is not stable'
            }
        
        # In a stable matching, no player has incentive to unilaterally deviate
        # However, we need to check if there's an incentive to deviate in the dynamic game
        
        # For a Nash equilibrium in the dynamic game:
        # 1. No player should have incentive to propose to someone else
        # 2. No player should have incentive to reject their current partner
        
        # This is a simplification - a full Nash equilibrium analysis in the dynamic game
        # would require checking all possible strategy profiles
        
        return {
            'is_nash_equilibrium': True,
            'type': 'Nash equilibrium in the static game',
            'note': 'This is a stable matching, which is a Nash equilibrium in the static game.'
        }
    
    def get_markov_perfect_equilibrium_analysis(self):
        """
        Analyze whether the current state could be part of a Markov Perfect Equilibrium.
        
        Returns:
            Dictionary with analysis of the current state as a potential MPE
        """
        # For a Markov Perfect Equilibrium:
        # 1. Strategies should depend only on the current state (not the history)
        # 2. Strategies should be a Nash equilibrium in all subgames
        
        # In the marriage game context, an MPE would typically involve:
        # - Players proposing to their best available partner
        # - Players accepting their best proposal
        # - Once a stable matching is reached, no more proposals/changes
        
        if self.is_stable():
            return {
                'could_be_mpe': True,
                'type': 'Markov Perfect Equilibrium',
                'note': 'The current stable matching could be part of an MPE where players play best-response strategies based on the current state.'
            }
        else:
            return {
                'could_be_mpe': False,
                'reason': 'Current matching is not stable, so it cannot be part of an MPE.'
            }
    
    def simulate_gale_shapley_dynamics(self, max_turns=100):
        """
        Simulate the game dynamics using Gale-Shapley-inspired strategies.
        Men propose and women dispose until a stable matching is reached.
        """
        # Critical fix: Initialize an empty matching (everyone is single)
        self.couples = {}
        
        # Initialize free men (all men start free)
        free_men = list(range(self.n))
        
        # Track which woman each man has next in his list
        next_proposal = {m: 0 for m in range(self.n)}
        
        # Current engagements (will be updated throughout the algorithm)
        engagements = {}  # woman -> man
        
        turn = 0
        
        # Continue until all men are matched or we reach max_turns
        while free_men and turn < max_turns:
            m = free_men[0]  # Get the first free man
            
            # If m has proposed to all women, he remains single
            if next_proposal[m] >= len(self.men_preferences[m]):
                free_men.pop(0)  # Remove m from free men
                continue
            
            # Get next woman to propose to
            w = self.men_preferences[m][next_proposal[m]]
            next_proposal[m] += 1
            
            # If w is free, she accepts the proposal
            if w not in engagements:
                engagements[w] = m
                free_men.pop(0)  # Remove m from free men
            else:
                # w is already engaged to some man
                m_prime = engagements[w]
                w_idx = w - self.n  # Convert to 0-based index for preference list
                
                # Check if w prefers m to her current engagement m_prime
                try:
                    rank_m = self.women_preferences[w_idx].index(m)
                    rank_m_prime = self.women_preferences[w_idx].index(m_prime)
                    
                    if rank_m < rank_m_prime:  # Lower index = higher preference
                        # w prefers m to m_prime
                        engagements[w] = m
                        free_men.pop(0)  # Remove m from free men
                        free_men.append(m_prime)  # Add m_prime back to free men
                    else:
                        # w prefers m_prime to m, m remains free
                        pass
                except ValueError:
                    # Handle case where one of the men is not in w's preference list
                    # Default to keeping current engagement
                    pass
            
            # Every 5 iterations or when free_men is empty, update the game state
            if turn % self.n == 0 or not free_men:
                # Convert engagements to couples format and update game state
                stage1_actions = {i: None for i in range(self.total_players)}
                stage2_actions = {i: None for i in range(self.total_players)}
                
                for w, m in engagements.items():
                    stage1_actions[m] = w
                    stage2_actions[w] = m
                    stage2_actions[m] = w
                
                # Play the turn to update utilities and game state
                self.play_turn(stage1_actions, stage2_actions)
            
            turn += 1
        
        # Final update if we haven't done one recently
        if engagements and (turn % 5 != 0):
            stage1_actions = {i: None for i in range(self.total_players)}
            stage2_actions = {i: None for i in range(self.total_players)}
            
            for w, m in engagements.items():
                stage1_actions[m] = w
                stage2_actions[w] = m
                stage2_actions[m] = w
            
            self.play_turn(stage1_actions, stage2_actions)
        
        return {
            'history': self.history,
            'final_state': {
                'couples': self.couples,
                'is_stable': self.is_stable(),
                'utilities': self.player_utilities
            }
        }
            
    def reset(self):
        """Reset the game state"""
        self.couples = {}
        self.turn = 0
        self.history = []
        self.player_utilities = {i: 0 for i in range(self.total_players)}


def load_preference_lists(json_path, list_index=0):
    """
    Load preference lists from a JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if list_index >= len(data):
        raise ValueError(f"List index {list_index} out of range (0-{len(data)-1})")
        
    return {
        "men_preferences": data[list_index]["men_preferences"],
        "women_preferences": data[list_index]["women_preferences"]
    }


def format_matching(couples, n):
    """Format matching for display"""
    result = "Current Matching:\n"
    result += "Player\tPartner\n"
    
    for i in range(n * 2):
        if i < n:
            player_type = "Man"
            idx = i
        else:
            player_type = "Woman"
            idx = i - n
            
        if i in couples:
            partner = couples[i]
            if partner < n:
                partner_type = "Man"
                partner_idx = partner
            else:
                partner_type = "Woman"
                partner_idx = partner - n
                
            result += f"{player_type} {idx}\t{partner_type} {partner_idx}\n"
        else:
            result += f"{player_type} {idx}\tSingle\n"
            
    return result


def analyze_fairness(game):
    """Analyze the fairness of the current matching"""
    if not game.couples:
        return "No matching to analyze."
        
    utilities = []
    men_utilities = []
    women_utilities = []
    
    for player, partner in game.couples.items():
        utility = game.f[player].get(partner, 0)
        utilities.append(utility)
        
        if player < game.n:
            men_utilities.append(utility)
        else:
            women_utilities.append(utility)
            
    if not utilities:
        return "No valid couples to analyze."
        
    result = "Fairness Analysis:\n"
    result += f"Minimum utility: {min(utilities):.4f}\n"
    result += f"Average utility: {sum(utilities) / len(utilities):.4f}\n"
    
    if men_utilities:
        result += f"Men's average utility: {sum(men_utilities) / len(men_utilities):.4f}\n"
    else:
        result += "Men's average utility: N/A\n"
        
    if women_utilities:
        result += f"Women's average utility: {sum(women_utilities) / len(women_utilities):.4f}\n"
    else:
        result += "Women's average utility: N/A\n"
    
    return result


def main():
    """Main function to demonstrate the algorithm"""
    if len(sys.argv) < 2:
        print("Usage: python dynamic_smp_game.py <json_file> [list_index=0]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    list_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    try:
        preferences = load_preference_lists(json_path, list_index)
        print(f"Loaded preference list {list_index} from {json_path}")
    except Exception as e:
        print(f"Error loading preference lists: {e}")
        sys.exit(1)
    
    n = len(preferences["men_preferences"])
    game = DynamicMarriageGame(n, preferences)
    
    print("Simulating Gale-Shapley dynamics...")
    result = game.simulate_gale_shapley_dynamics()
    
    print("\nFinal state after", game.turn, "turns:")
    print(format_matching(game.couples, n))
    
    print("\nIs the matching stable?", game.is_stable())
    
    nash_analysis = game.get_nash_equilibrium_strategy()
    print("\nNash Equilibrium Analysis:")
    for key, value in nash_analysis.items():
        print(f"{key}: {value}")
        
    mpe_analysis = game.get_markov_perfect_equilibrium_analysis()
    print("\nMarkov Perfect Equilibrium Analysis:")
    for key, value in mpe_analysis.items():
        print(f"{key}: {value}")
    
    print("\n" + analyze_fairness(game))
    
    print("\nFinal Utilities:")
    for player, utility in game.player_utilities.items():
        if player < n:
            player_type = "Man"
            idx = player
        else:
            player_type = "Woman"
            idx = player - n
            
        print(f"{player_type} {idx}: {utility:.4f}")


if __name__ == "__main__":
    main()