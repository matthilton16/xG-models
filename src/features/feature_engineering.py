import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

class FeatureEngineer:
    def __init__(self, pitch_length_m: float = 100.0, pitch_width_m: float = 100.0, goal_width_m: float = 7.32):
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.goal_width_m = goal_width_m
        self.goal_posts_y_m = (-self.goal_width_m / 2, self.goal_width_m / 2)
        self.goal_line_x_m = self.pitch_length_m / 2 
        print(f"FeatureEngineer initialized with pitch (LxW): {pitch_length_m}m x {pitch_width_m}m, Goal width: {goal_width_m}m")

    def _calculate_tracking_features(self, shot_event_t_frame_id: str, 
                                               tracking_df: pd.DataFrame) -> dict:
        features = {}
        shooter_data = tracking_df[tracking_df['shooter'] == True]
        if shooter_data.empty: return {k: np.nan for k in self._get_adv_feature_keys()} # Return NaNs if no shooter
        
        # Shooter to GK relative distance 
        shooter_pos_x = shooter_data['pos_x'].iloc[0]
        shooter_pos_y = shooter_data['pos_y'].iloc[0]

        gk_data = tracking_df[
            (tracking_df['Goalkeeper'] == 'Yes') & 
            (tracking_df['team_status'] == 'defending')
        ]
        if gk_data.empty: return {k: np.nan for k in self._get_adv_feature_keys()} 

        gk_pos_x = gk_data['pos_x'].iloc[0]
        gk_pos_y = gk_data['pos_y'].iloc[0]

        features['gk_distance_m'] = np.sqrt((shooter_pos_x - gk_pos_x)**2 + (shooter_pos_y - gk_pos_y)**2)
        features['gk_distance_y_m'] = np.abs(shooter_pos_y - gk_pos_y)
        features['gk_dist_to_goal_m'] = np.sqrt((self.goal_line_x_m - gk_pos_x)**2 + (0 - gk_pos_y)**2)

        # Defender Proximity 
        defender_positions = tracking_df[
            (tracking_df['team_status'] == 'defending') & (tracking_df['Goalkeeper'] == 'No')
        ][['pos_x', 'pos_y']]
        if not defender_positions.empty:
            distances_to_shooter = np.sqrt((shooter_pos_x - defender_positions['pos_x'])**2 + (shooter_pos_y - defender_positions['pos_y'])**2)
            features['close_players'] = np.sum(distances_to_shooter < 3.0)
        else:
            features['close_players'] = 0

        # Triangle Opponent and Teammate 
        v_shooter = np.array([shooter_pos_x, shooter_pos_y])
        v_post1 = np.array([self.goal_line_x_m, self.goal_posts_y_m[0]])
        v_post2 = np.array([self.goal_line_x_m, self.goal_posts_y_m[1]])

        features['triangle_opp'] = 0
        features['triangle_tm'] = 0
        for _, player_row in tracking_df.iterrows():
            if player_row['shooter']: continue
            if self._is_in_triangle(player_row['pos_x'], player_row['pos_y'], v_shooter, v_post1, v_post2):
                if player_row['team_status'] == 'defending' and player_row['Goalkeeper'] == 'No': # Non-GK defenders
                    features['triangle_opp'] += 1
                elif player_row['team_status'] == 'attacking':
                    features['triangle_tm'] += 1
        
        # Shooter Velocity and Acceleration 
        features['vx'] = shooter_data['speed_x'].iloc[0] if 'speed_x' in shooter_data.columns and not shooter_data['speed_x'].empty else 0
        features['vy'] = shooter_data['speed_y'].iloc[0] if 'speed_y' in shooter_data.columns and not shooter_data['speed_y'].empty else 0
        features['acc'] = shooter_data['acc'].iloc[0] if 'acc' in shooter_data.columns and not shooter_data['acc'].empty else 0
        
        
        return features
    
    def _get_adv_feature_keys(self):
        # Helper to get keys for consistent NaN dict creation
        return ['gk_distance_m', 'gk_distance_y_m', 'gk_dist_to_goal_m', 
                'close_players', 'triangle_opp', 'triangle_tm', 'vx', 'vy', 'acc']

    def _is_in_triangle(self, px, py, v0, v1, v2):
        p = np.array([px, py])
        # Using cross product method for point in triangle test
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, v0, v1)
        d2 = sign(p, v1, v2)
        d3 = sign(p, v2, v0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos) 
    
    def generate_features(self, event_df: pd.DataFrame, 
                          tracking_df: pd.DataFrame,
                          event_coord_pitch_length: float = 100.0, 
                          event_coord_pitch_width: float = 100.0) -> tuple[pd.DataFrame, list[str]]:
        print("Starting full feature generation pipeline...")
        
        # Start feature calculation
        print("Calculating basic and advanced features per shot...")
        all_features_list = []

        for original_idx, shot_event_row in event_df.iterrows():

            features = {'original_index': original_idx}
            features['t_frame_id'] = shot_event_row['t_frame_id']

            # Add player identifier
            features['player'] = shot_event_row['player_id']
            
            # Body Part
            features['foot'] = 1 if isinstance(shot_event_row['body part'], str) and shot_event_row['body part'].lower() in ['left foot', 'right foot'] else 0
            features['head'] = 1 if isinstance(shot_event_row['body part'], str) and shot_event_row['body part'].lower() in ['head', 'other body part', 'other'] else 0

            # Location
            event_x = pd.to_numeric(shot_event_row['x'], errors='coerce')
            event_y = pd.to_numeric(shot_event_row['y'], errors='coerce')
            features['x0_event'] = event_x
            features['y0_event'] = event_y

            # Angle and Distance
            x_dist_to_goal_line_event_scaled = (event_coord_pitch_length - event_x) * (self.pitch_length_m / event_coord_pitch_length)
            y_dist_from_center_event_scaled = np.abs(event_y - event_coord_pitch_width / 2) * (self.pitch_width_m / event_coord_pitch_width)

            x_calc = x_dist_to_goal_line_event_scaled
            c_calc = y_dist_from_center_event_scaled
            
            current_angle = np.nan
            current_distance_m = np.nan

            if pd.notna(x_calc) and pd.notna(c_calc):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    denominator = x_calc**2 + c_calc**2 - (self.goal_width_m / 2)**2
                    safe_denominator = np.where(denominator == 0, 1e-6, denominator)
                    angle_rad_arg = self.goal_width_m * x_calc / safe_denominator
                    angle_rad = np.arctan(angle_rad_arg)
                    current_angle = np.where(angle_rad >= 0, angle_rad * 180 / np.pi, (angle_rad + np.pi) * 180 / np.pi)
                current_distance_m = np.sqrt(x_calc**2 + c_calc**2)

            features['angle'] = current_angle if pd.notna(current_angle) else 0
            features['distance_m'] = current_distance_m if pd.notna(current_distance_m) else self.pitch_length_m

            #outcome
            features['goal'] = shot_event_row['outcome']

            # tracking-based features
            adv_feats_for_shot = {k: np.nan for k in self._get_adv_feature_keys()} # Init with NaNs
            t_frame_id = shot_event_row['t_frame_id']
            if t_frame_id in tracking_df['t_frame_id'].unique():
                shot_frame_tracking_data = tracking_df[tracking_df['t_frame_id'] == t_frame_id]
                adv_feats_for_shot.update(self._calculate_tracking_features(t_frame_id, shot_frame_tracking_data))

            features.update(adv_feats_for_shot)
            all_features_list.append(features)
        
        # Model variables
        features_df = pd.DataFrame(all_features_list).set_index('original_index')
        
        X_feature_names = [
            "x0_event", "y0_event", "angle", "distance_m", "gk_distance_m", 
            "gk_distance_y_m", "triangle_opp", "triangle_tm", "close_players", 
            "head", "foot",
            'vx', 'vy', 'acc'
        ]
        # Select only specified features and target, and ensure they exist
        feature_cols = []
        for f_name in X_feature_names:
            if f_name in features_df.columns:
                feature_cols.append(f_name)
            else:
                print(f"Warning: Feature '{f_name}' expected but not found in model_vars. It will be missing from output.")
        
        # Add target variable 'goal', 't_frame_id' and 'player' for reference
        if 'goal' in features_df.columns: feature_cols.append('goal')
        if 't_frame_id' in features_df.columns: feature_cols.append('t_frame_id')
        if 'player' in features_df.columns: feature_cols.append('player') # Ensure player is in final output
        
        features_df = features_df[list(dict.fromkeys(feature_cols))].copy() 

        # Drop rows if any of the selected X_feature_names are NaN
        cols_for_nan_check = [f for f in X_feature_names if f in features_df.columns]
        if cols_for_nan_check:
            features_df.dropna(subset=cols_for_nan_check, inplace=True)
        
        print(f"Final model_vars shape after NaN drop and selection: {features_df.shape}")
        # Return only the X_feature_names that are actually present in the final_model_vars
        present_X_feature_names = [f for f in X_feature_names if f in features_df.columns]
        return features_df, present_X_feature_names

    def split_and_scale_data(self, features_df: pd.DataFrame, X_feature_names: list[str], 
                             y_col_name: str = 'goal', train_size: float = 0.7, 
                             cal_proportion_of_holdout: float = 0.5,
                             random_state: int = 42,
                             train_test_only: bool = False) -> tuple:
        print("Splitting and scaling data...")
        
        if features_df.empty or y_col_name not in features_df.columns or not X_feature_names:
            print("Warning: features_df is empty, target column missing, or no feature names provided. Cannot split/scale.")
            return (np.array([]), np.array([]), np.array([]), pd.Series(dtype='int'), pd.Series(dtype='int'), pd.Series(dtype='int'), StandardScaler())

        valid_X_features = [f for f in X_feature_names if f in features_df.columns]
        if not valid_X_features:
            print("Warning: No valid X_feature_names found in features_df. Cannot split/scale.")
            return (np.array([]), np.array([]), np.array([]), pd.Series(dtype='int'), pd.Series(dtype='int'), pd.Series(dtype='int'), StandardScaler())
        
        X = features_df[valid_X_features]
        y = features_df[y_col_name]

        if X.empty:
            print("Warning: Feature set X is empty after selection. Cannot split and scale.")
            return (np.array([]), np.array([]), np.array([]), pd.Series(dtype='int'), pd.Series(dtype='int'), pd.Series(dtype='int'), StandardScaler())

        can_stratify_y = y.nunique() > 1 and all(y.value_counts(dropna=False) >= 2)
        stratify_main = y if can_stratify_y else None

        X_train, X_temp_holdout, y_train, y_temp_holdout = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=stratify_main
        )

        # Initialize val and cal sets, to be populated based on train_test_only flag
        X_val, y_val = pd.DataFrame(columns=valid_X_features), pd.Series(dtype=y.dtype, name=y_col_name)
        X_cal, y_cal = pd.DataFrame(columns=valid_X_features), pd.Series(dtype=y.dtype, name=y_col_name)
        X_test, y_test = pd.DataFrame(columns=valid_X_features), pd.Series(dtype=y.dtype, name=y_col_name)


        if train_test_only:
            if not X_temp_holdout.empty:
                X_test = X_temp_holdout
                y_test = y_temp_holdout
            # X_val, y_val, X_cal, y_cal remain empty as initialized
        else: # Original 3-way split logic
            if not X_temp_holdout.empty:
                can_stratify_temp_holdout = y_temp_holdout.nunique() > 1 and all(y_temp_holdout.value_counts(dropna=False) >=2)
                stratify_temp_holdout = y_temp_holdout if can_stratify_temp_holdout else None
                
                # X_cal gets cal_proportion_of_holdout, remainder is X_val
                X_cal_split, X_val_split, y_cal_split, y_val_split = train_test_split(
                    X_temp_holdout, y_temp_holdout, train_size=cal_proportion_of_holdout, 
                    random_state=random_state, stratify=stratify_temp_holdout
                )
                X_cal, y_cal = X_cal_split, y_cal_split
                X_val, y_val = X_val_split, y_val_split
            # If X_temp_holdout is empty, X_val, y_val, X_cal, y_cal remain empty as initialized.

        scaler = StandardScaler()
        X_train_scaled = np.array([])
        if not X_train.empty:
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            print("Warning: Training set X_train is empty. Scaler not fitted on data.")
            scaler.fit(pd.DataFrame(columns=X.columns)) 
        
        X_val_scaled = np.array([])
        X_cal_scaled = np.array([])

        if train_test_only:
            if not X_test.empty and hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                X_val_scaled = scaler.transform(X_test) # X_val_scaled now holds scaled test data
            y_val = y_test # y_val now holds test targets
            # X_cal_scaled remains empty np.array, y_cal remains empty Series
        else: # 3-way split
            if not X_val.empty and hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                 X_val_scaled = scaler.transform(X_val)
            if not X_cal.empty and hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                X_cal_scaled = scaler.transform(X_cal)
        
        # Ensure consistent return types (numpy arrays for X, pandas Series for y)
        X_train_scaled = np.array(X_train_scaled) if not isinstance(X_train_scaled, np.ndarray) else X_train_scaled
        X_val_scaled = np.array(X_val_scaled) if not isinstance(X_val_scaled, np.ndarray) else X_val_scaled
        X_cal_scaled = np.array(X_cal_scaled) if not isinstance(X_cal_scaled, np.ndarray) else X_cal_scaled
        
        y_train = pd.Series(y_train, name=y_col_name, dtype=y.dtype) if not isinstance(y_train, pd.Series) else y_train.astype(y.dtype)
        y_val = pd.Series(y_val, name=y_col_name, dtype=y.dtype) if not isinstance(y_val, pd.Series) else y_val.astype(y.dtype)
        y_cal = pd.Series(y_cal, name=y_col_name, dtype=y.dtype) if not isinstance(y_cal, pd.Series) else y_cal.astype(y.dtype)

        if train_test_only:
            print(f"Data split (Train/Test only): Train ({X_train_scaled.shape[0]}), Test ({X_val_scaled.shape[0]})")
        else:
            print(f"Data split (Train/Val/Cal): Train ({X_train_scaled.shape[0]}), Validation ({X_val_scaled.shape[0]}), Calibration ({X_cal_scaled.shape[0]})")
        return X_train_scaled, X_val_scaled, X_cal_scaled, y_train, y_val, y_cal, scaler
