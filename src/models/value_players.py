import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

class PlayerValueCalculator:
    def __init__(self, player_shot_data: pd.DataFrame):
        """
        Initializes the PlayerValueCalculator with shot-level data for players.

        Args:
            player_shot_data (pd.DataFrame): DataFrame with columns like 
                                             ['player', 'goal', 'xG_mean', 'xG_variance'].
                                             'player': Identifier for the player.
                                             'goal': Binary outcome of the shot (1 for goal, 0 otherwise).
                                             'xG_mean': Predicted xG (expected goals) for the shot.
                                             'xG_variance': Variance associated with the xG prediction.
        """
        if not all(col in player_shot_data.columns for col in ['player', 'goal', 'xG_mean', 'xG_variance']):
            raise ValueError("Input DataFrame must contain columns: 'player', 'goal', 'xG_mean', 'xG_variance'")
        self.player_shot_data = player_shot_data.copy()
        print(f"PlayerValueCalculator initialized with {len(self.player_shot_data)} shots.")

    def calculate_player_summary(self, min_shots_threshold: int = 15) -> pd.DataFrame:
        """
        Calculates a summary for each player based on their shots, including average value..
        Filters players by a minimum number of shots.
        """
        print(f"Calculating player summary with min_shots_threshold = {min_shots_threshold}...")
        # Filter players by min_shots_threshold
        player_counts = self.player_shot_data.groupby('player')['goal'].count()
        players_above_threshold = player_counts[player_counts > min_shots_threshold].index
        
        if players_above_threshold.empty:
            print("No players meet the minimum shot threshold.")
            return pd.DataFrame()
            
        filtered_data = self.player_shot_data[self.player_shot_data['player'].isin(players_above_threshold)]

        # Aggregations
        player_summary = filtered_data.groupby('player').agg(
            no_of_shots=('goal', 'count'),
            sum_xG_mean=('xG_mean', 'sum'),
            sum_goals=('goal', 'sum'),
            avg_xG_mean=('xG_mean', 'mean'),
            avg_xG_variance=('xG_variance', 'mean'),
            avg_goals=('goal', 'mean')
        ).reset_index()

        player_summary['avg_Value'] = player_summary['avg_goals'] - player_summary['avg_xG_mean']
        player_summary = player_summary.sort_values('avg_Value', ascending=False)
        print("Player summary calculation complete.")
        return player_summary

    def calculate_soft_weighted_value(self) -> pd.DataFrame:
        """
        Calculates player value using a variance soft-weighting method.
        Shots with lower variance (more certainty in xG) get higher weight.
        """
        print("Calculating soft-weighted player value...")
        data = self.player_shot_data.copy()
        data['value_shot'] = data['goal'] - data['xG_mean']
        
        # Calculate weight, avoid division by zero or extremely large weights if variance is tiny
        data['weight'] = 1 / np.maximum(data['xG_variance'], 1e-9) # Add epsilon to variance

        # Normalize weights per player
        data['normalized_weight'] = data.groupby('player')['weight'].transform(lambda x: x / x.sum())
        data['weighted_value_shot'] = data['normalized_weight'] * data['value_shot']

        result = data.groupby('player').agg(
            total_weighted_value=('weighted_value_shot', 'sum'),
            no_goals=('goal', 'sum'),
            no_shots=('goal', 'count') # Renamed from player to no_shots for clarity
        ).reset_index()

        result = result.sort_values(by='total_weighted_value', ascending=False)
        print("Soft-weighted value calculation complete.")
        return result
    
    def plot_player_average_value(self, player_summary_df: pd.DataFrame, top_n: int = 20, 
                                     show_plot: bool = True, save_path: str = None):
        """Plots average value with confidence intervals for top N players."""
        if player_summary_df.empty or not all(col in player_summary_df.columns for col in ['player', 'avg_Value']):
            print("Player summary DataFrame is invalid or missing columns for plot.")
            return

        print(f"Plotting average value for top {top_n} players...")
        plot_df = player_summary_df.nlargest(top_n, 'avg_Value')
        player_names = plot_df['player']
        avg_values = plot_df['avg_Value']

        fig, ax = plt.subplots(figsize=(12, 8))
        x_pos = np.arange(len(player_names))
        ax.bar(x_pos, avg_values, align='center', alpha=0.7, ecolor='black', capsize=5, color='skyblue')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(player_names, rotation=45, ha="right")
        ax.set_ylabel('Average Value (Goals - xG_mean per shot)')
        ax.set_title(f'Top {top_n} Players by Average Value')
        ax.yaxis.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Player Values plot saved to {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()

class BayesianPlayerMarginalization:
    """
    Calculates marginalized goal probability P(goal | player) by integrating over shot features.
    
    This addresses the core question: "What's a player's true shooting ability when we account
    for the difficulty distribution of shots they face?"
    """
    
    def __init__(self, player_valuation_input_df: pd.DataFrame, min_shots: int = 15):
        """
        Initialize with the complete dataset containing features and outcomes.
        
        Args:
            player_valuation_input_df: DataFrame with columns:
                - 'player': Player identifier
                - 'goal': Binary outcome (1 for goal, 0 for miss)  
                - 'xG_mean': Expected goal probability from your xG model
                - 'xG_variance': Variance from your xG model
                - Shot features: x0_event, y0_event, angle, distance_m, gk_distance_m,
                  gk_distance_y_m, triangle_opp, triangle_tm, close_players, head, foot, vx, vy, acc
            min_shots: Minimum shots required to include a player
        """
        self.data = player_valuation_input_df.copy()
        self.min_shots = min_shots
        
        # Define the specific feature columns for football shots
        self.feature_columns = [
            "x0_event", "y0_event", "angle", "distance_m", "gk_distance_m", 
            "gk_distance_y_m", "triangle_opp", "triangle_tm", "close_players", 
            "head", "foot", 'vx', 'vy', 'acc'
        ]
        self.core_columns = ['player', 'goal', 'xG_mean', 'xG_variance']
        
        # Remove players with insufficient shots
        player_counts = self.data.groupby('player').size()
        valid_players = player_counts[player_counts >= min_shots].index
        self.data = self.data[self.data['player'].isin(valid_players)]
        
        # Initialize components
        self.scaler = StandardScaler()
        self.player_feature_models = {}
        self.nn_model = None  # For finding similar shots
        
        print(f"Initialized with {len(self.data)} shots from {len(valid_players)} players")
        print(f"Using {len(self.feature_columns)} features: {self.feature_columns[:5]}...")
        
    def fit_player_feature_distributions(self, n_components: int = 3):
        """
        Learn P(features | player) for each player using Gaussian Mixture Models.
        
        Args:
            n_components: Number of Gaussian components per player (default 3)
        """
        print("Learning feature distributions for each player using Gaussian Mixture Models...")
        
        # Prepare and standardize features - handle missing values
        feature_data = self.data[self.feature_columns].fillna(self.data[self.feature_columns].median())
        self.standardized_features = self.scaler.fit_transform(feature_data)
        
        # Fit nearest neighbors for xG lookup (using standardized features)
        self.nn_model = NearestNeighbors(n_neighbors=50, metric='euclidean')
        self.nn_model.fit(self.standardized_features)
        
        # Fit Gaussian Mixture Model for each player
        successful_fits = 0
        for player in self.data['player'].unique():
            player_mask = self.data['player'] == player
            player_features = self.standardized_features[player_mask]
            
            if len(player_features) < 5:
                continue
                
            try:
                # Adjust number of components based on available data
                n_comp = min(n_components, len(player_features) // 4, 5)
                n_comp = max(1, n_comp)  # At least 1 component
                
                model = GaussianMixture(
                    n_components=n_comp, 
                    covariance_type='full',
                    random_state=42,
                    max_iter=200,
                    tol=1e-3
                )
                model.fit(player_features)
                self.player_feature_models[player] = model
                successful_fits += 1
                
            except Exception as e:
                print(f"Warning: Could not fit model for {player}: {e}")
                continue
        
        print(f"Successfully fitted Gaussian Mixture Models for {successful_fits} players")
        return self
    
    def calculate_marginalized_probabilities(self, n_samples: int = 1000):
        """
        Calculate P(goal | player) by marginalizing over shot features.
        
        Implementation of: P(goal | player) = ∫ P(goal | features) × P(features | player) d(features)
        
        Uses the xG_mean values from your existing xG model for P(goal | features).
        
        Args:
            n_samples: Number of Monte Carlo samples per player
        """
        print(f"Calculating marginalized probabilities using {n_samples} samples per player...")
        
        results = []
        
        for player in self.player_feature_models.keys():
            player_data = self.data[self.data['player'] == player]
            
            # Sample from P(features | player)
            feature_samples = self._sample_player_features(player, n_samples)
            
            # Calculate P(goal | features) using existing xG model predictions
            goal_probs = self._predict_goals_using_xg_lookup(feature_samples)
            
            # Marginalize: average over all sampled feature combinations
            marginalized_prob = np.mean(goal_probs)
            
            # Calculate performance metrics
            actual_conversion = player_data['goal'].mean()
            expected_conversion = player_data['xG_mean'].mean()
            shot_difficulty = player_data['xG_mean'].std()  # Variability in shot difficulty
            
            results.append({
                'player': player,
                'marginalized_probability': marginalized_prob,
                'actual_conversion_rate': actual_conversion,
                'expected_conversion_rate': expected_conversion,
                'skill_above_marginalized': actual_conversion - marginalized_prob,
                'skill_above_expected_xg': actual_conversion - expected_conversion,
                'shot_difficulty_variance': shot_difficulty,
                'total_shots': len(player_data)
            })
        
        # Create results DataFrame and rank players
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('skill_above_marginalized', ascending=False)
        results_df['marginalized_rank'] = range(1, len(results_df) + 1)
        
        print(f"Marginalization complete! Ranked {len(results_df)} players")
        return results_df
    
    def _sample_player_features(self, player: str, n_samples: int) -> np.ndarray:
        """Sample feature vectors from the player's learned distribution."""
        model = self.player_feature_models[player]
        
        if hasattr(model, 'sample'):  # Gaussian Mixture Model
            samples, _ = model.sample(n_samples)
        else:  # Simple Gaussian
            samples = np.random.multivariate_normal(
                model['mean'], 
                model['cov'], 
                size=n_samples
            )
        
        return samples
    
    def _predict_goals_using_xg_lookup(self, feature_samples: np.ndarray) -> np.ndarray:
        """
        Use existing xG_mean values by finding similar shots in the dataset.
        This leverages your already-trained xG model predictions.
        """
        goal_probs = []
        
        for features in feature_samples:
            # Find nearest neighbors in standardized feature space
            distances, indices = self.nn_model.kneighbors([features], n_neighbors=20)
            
            # Use inverse distance weighting to combine xG_mean values from similar shots
            weights = 1 / (distances[0] + 1e-8)  # Small epsilon to avoid division by zero
            weights = weights / weights.sum()  # Normalize weights
            
            # Get xG_mean values from nearest shots
            nearest_xg_values = self.data.iloc[indices[0]]['xG_mean'].values
            
            # Calculate weighted average xG
            weighted_xg = np.average(nearest_xg_values, weights=weights)
            goal_probs.append(weighted_xg)
        
        return np.array(goal_probs)
    
    def _predict_goals_empirically(self, feature_samples: np.ndarray) -> np.ndarray:
        """
        Empirical approach: find similar shots and use their actual conversion rate.
        """
        goal_probs = []
        
        for features in feature_samples:
            # Find nearest neighbors
            distances, indices = self.nn_model.kneighbors([features], n_neighbors=30)
            
            # Use actual goal outcomes of similar shots
            similar_outcomes = self.data.iloc[indices[0]]['goal'].values
            conversion_rate = similar_outcomes.mean()
            
            goal_probs.append(conversion_rate)
        
        return np.array(goal_probs)
    
    def get_player_shooting_profile(self, player: str, n_samples: int = 100) -> pd.DataFrame:
        """
        Analyze a specific player's shooting profile by sampling from their distribution.
        """
        if player not in self.player_feature_models:
            raise ValueError(f"Player {player} not found in fitted models")
        
        # Sample from player's feature distribution
        samples = self._sample_player_features(player, n_samples)
        original_samples = self.scaler.inverse_transform(samples)
        
        # Create DataFrame with feature names
        profile_df = pd.DataFrame(original_samples, columns=self.feature_columns)
        profile_df['player'] = player
        
        # Add summary statistics
        actual_player_data = self.data[self.data['player'] == player][self.feature_columns]
        
        summary = {
            'feature': self.feature_columns,
            'actual_mean': actual_player_data.mean().values,
            'actual_std': actual_player_data.std().values,
            'sampled_mean': profile_df[self.feature_columns].mean().values,
            'sampled_std': profile_df[self.feature_columns].std().values
        }
        
        return profile_df, pd.DataFrame(summary)
    
    def compare_players(self, player1: str, player2: str) -> dict:
        """
        Compare two players' shooting profiles and marginalized probabilities.
        """
        if player1 not in self.player_feature_models or player2 not in self.player_feature_models:
            raise ValueError("One or both players not found in fitted models")
        
        comparison = {}
        
        for player in [player1, player2]:
            player_data = self.data[self.data['player'] == player]
            comparison[player] = {
                'actual_conversion': player_data['goal'].mean(),
                'expected_conversion': player_data['xG_mean'].mean(),
                'shots_taken': len(player_data),
                'avg_shot_difficulty': player_data['xG_mean'].mean(),
                'shot_difficulty_variance': player_data['xG_mean'].std()
            }
        
        return comparison


    