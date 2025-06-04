import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

class DataProcessor:
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        # Set-up directories
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)   
        self.path_tracking = self.raw_data_dir / 'tracking'
        self.path_events = self.raw_data_dir / 'events'

        # Load event names
        event_names_path = self.processed_data_dir / 'event_names.csv'
        df_event_names = pd.read_csv(event_names_path)
        self.dict_event_names = df_event_names.set_index('event_type_id').to_dict()['event_description']

        # Load qualifier names 
        qualifier_names_path = self.processed_data_dir / 'qualifier_names.csv'
        df_qualifier_names = pd.read_csv(qualifier_names_path)
        self.dict_qualifier_names = df_qualifier_names.set_index('qualifierId').to_dict()['qualifier']

        # Shot dictionaries
        self.shot_dict = {13: 'Shot off target', 14: 'Post', 15: 'Shot saved', 16: 'Goal'}
        self.body_dict = {15: "head", 72: "left foot", 20: "right foot", 21: "other body part"}
        self.shot_play_dict = {
            22: 'regular play', 23: 'fast break', 24: 'set piece', 25: 'from corner',
            26: 'free kick', 96: 'corner situation', 112: 'scramble',
            160: 'throw-in set piece', 9: 'penalty'
        }

    def _add_timeelapsed_to_events_df(self, df_group, file_name: str):
        """
        Adds a timeelapsed column to a dataframe.
        """
        period_id_for_warning = df_group.name 
       
        start_time_df = df_group[df_group['typeId'] == 32]
        if start_time_df.empty:
            df_group['timestamp_new'] = 0 
            df_group['timeelapsed'] = 0.0
            df_group['periodId'] = period_id_for_warning 
            return df_group

        start_time_dt = pd.to_datetime(start_time_df['timeStamp'].iloc[0], errors='coerce')
        if pd.isna(start_time_dt):
            df_group['timestamp_new'] = 0
            df_group['timeelapsed'] = 0.0
            df_group['periodId'] = period_id_for_warning 
            return df_group
        
        start_time = start_time_dt.timestamp()

        # Initialize columns for all rows
        df_group['timestamp_new'] = 0
        df_group['timeelapsed'] = 0.0

        # Attempt to parse all timestamps 
        parsed_timestamps = pd.to_datetime(df_group['timeStamp'], errors='coerce')
        valid_mask = parsed_timestamps.notna()

        # Calculate valid timestamps
        if valid_mask.any():
            timestamp_vals = parsed_timestamps[valid_mask].apply(lambda x: x.timestamp())
            df_group.loc[valid_mask, 'timestamp_new'] = np.int64((timestamp_vals - start_time) * 1000)
            df_group.loc[valid_mask, 'timeelapsed'] = df_group.loc[valid_mask, 'timestamp_new'].apply(lambda x: (40 * round(x / 40)) / 1000)
        
        df_group['periodId'] = period_id_for_warning 
        return df_group

    def _load_single_event_json(self, file_name: str):
        """
        Loads a single event JSON file and returns a dataframe.
        """
        with open(self.path_events / file_name) as f:
            data = json.loads(f.read())
        f.close()
        
        events_df = pd.json_normalize(data['liveData']['event'])
        
        required_cols = ['typeId','contestantId','periodId','timeMin', 'timeSec','timeStamp','playerId','outcome', 'qualifier', 'x', 'y']
        events_df = events_df[[col for col in required_cols if col in events_df.columns]].copy()
        
        events_df['timestamp_dt'] = pd.to_datetime(events_df.timeStamp, errors='coerce')
        events_df = events_df[events_df['periodId'].isin([1, 2])].copy() 

        events_df = events_df.groupby('periodId', group_keys=False).apply(self._add_timeelapsed_to_events_df, include_groups=False, file_name=file_name)
        
        events_df = events_df.drop(columns=['timeStamp', 'timestamp_new'], errors='ignore')
       
        events_df = events_df.rename(columns={
            'periodId': 'current_phase', 'typeId': 'event_type_id',
            'timeMin': 'period_minute', 'timeSec': 'period_second'
        })
        
        return events_df
    
    def _consolidate_event_json_files(self, output_filename="consolidated_events.parquet"):
        """
        Loads all raw event JSON files, processes them using _load_single_event_json,
        concatenates them, and saves the result as a Parquet file in the raw_data_dir.
        """
        all_event_dfs = []
        json_files = [f for f in os.listdir(self.path_events) if f.endswith('.json') and not f.startswith('.DS_Store')]

        for i, json_file in enumerate(json_files):
            game_id = os.path.splitext(json_file)[0]
            print(f"  Processing event file {i+1}/{len(json_files)}: {json_file} (Game ID: {game_id})")
            try:
                df_event_game = self._load_single_event_json(json_file)
                if not df_event_game.empty:
                    if 'game_id' not in df_event_game.columns:
                         df_event_game['game_id'] = int(game_id)
                    all_event_dfs.append(df_event_game)
                else:
                    print(f"    Warning: No data returned from _load_single_event_json for {json_file}.")
            except Exception as e:
                print(f"    Error processing {json_file}: {e}")

        if not all_event_dfs:
            print(f"No event data successfully processed from any JSON file. {output_filename} will be empty or not created.")
            return pd.DataFrame()

        consolidated_df = pd.concat(all_event_dfs, ignore_index=True)
        output_path = self.raw_data_dir / output_filename
        
        try:
            consolidated_df.to_parquet(output_path, index=False)
            print(f"Consolidated event data saved to {output_path} ({len(consolidated_df)} rows)")
        except Exception as e:
            print(f"Error saving consolidated event data to {output_path}: {e}")
            return pd.DataFrame()
            
        return consolidated_df
    
    def create_shot_event_df(self, consolidated_events_filename="consolidated_events.parquet", output_shots_filename="shot_event_df.csv"):
        """
        Creates a shot event dataframe from the consolidated events dataframe.
        """
        consolidated_events_path = self.raw_data_dir / consolidated_events_filename

        consolidated_events_df = pd.read_parquet(consolidated_events_path)
        shots_df_source = consolidated_events_df[consolidated_events_df['event_type_id'].isin([13, 14, 15, 16])].copy()

        contestant_id_list, game_id_list, outcome_list, shot_name_list, shot_x_list, shot_y_list = [], [], [], [], [], []
        time_elapsed_list, shot_period_list, goalmouth_y_list, goalmouth_z_list = [], [], [], []
        saved_x_list, saved_y_list, body_part_list, shot_play_list, shot_player_list = [], [], [], [], []
        og_list = []
        goal_list = []

        for _, event in shots_df_source.iterrows():
            contestant_id_list.append(event.get('contestantId'))
            game_id_list.append(event.get('game_id'))
            shot_name_list.append(event.get('event_description'))
            shot_x_list.append(event.get('x'))
            shot_y_list.append(event.get('y'))
            time_elapsed_list.append(event.get('timeelapsed'))
            shot_period_list.append(event.get('current_phase'))
            shot_player_list.append(event.get('playerId'))
            og_list.append(0)

            outcome_list.append(event['outcome'] if event['event_type_id'] == 16 else 0)
            goal_list.append(1 if event['event_type_id'] == 16 else 0)
            
            current_goalmouth_y, current_goalmouth_z, current_saved_x, current_saved_y = '', '', '', ''
            current_body_part, current_shot_play = '', ''
            
            qualifiers = event.get('qualifier')
            if isinstance(qualifiers, np.ndarray):
                for q in qualifiers:
                    if not isinstance(q, dict): continue
                    qualifier_id = q.get("qualifierId")
                    value = q.get("value")
                    
                    if qualifier_id == 102: current_goalmouth_y = value
                    if qualifier_id == 103: current_goalmouth_z = value
                    if qualifier_id in self.body_dict: current_body_part = self.body_dict[qualifier_id]
                    if qualifier_id in self.shot_play_dict: current_shot_play = self.shot_play_dict[qualifier_id]
                    if qualifier_id == 27: og_list[-1] = 'own goal attempt-attacking team'
                    if qualifier_id == 28: og_list[-1] = 'own goal attempt-defending team'
                    if event['event_type_id'] == 15:
                        if qualifier_id == 146: current_saved_x = value
                        if qualifier_id == 147: current_saved_y = value

            goalmouth_y_list.append(current_goalmouth_y)
            goalmouth_z_list.append(current_goalmouth_z)
            saved_x_list.append(current_saved_x)
            saved_y_list.append(current_saved_y)
            body_part_list.append(current_body_part)
            shot_play_list.append(current_shot_play)

        shot_data = {
            "contestant_id": contestant_id_list, "game_id": game_id_list, "outcome": outcome_list,
            "player_id": shot_player_list, "current_phase": shot_period_list, "timeelapsed": time_elapsed_list,
            "shot play": shot_play_list, "shot type": shot_name_list, "body part": body_part_list,
            "x": shot_x_list, "y": shot_y_list, "goalmouth y": goalmouth_y_list,
            "goalmouth z": goalmouth_z_list, "saved x": saved_x_list, "saved y": saved_y_list, "og": og_list,
            "goal": goal_list
        }
        shot_event_df = pd.DataFrame(shot_data)
        shot_event_df = shot_event_df[shot_event_df['og'] != 'own goal attemp-defending team'].copy()
        shot_event_df.drop(columns=['og'], inplace=True, errors='ignore')

        #Create t_frame_id identifier for merging with tracking data
        shot_event_df.loc[:, 't_frame_id'] = (shot_event_df.loc[:,'game_id'].astype(str) + '_' + shot_event_df.loc[:,'current_phase'].astype(str) + '_' + shot_event_df.loc[:,'timeelapsed'].astype(str))

        output_path = self.processed_data_dir / output_shots_filename
        shot_event_df.to_csv(output_path, index=False) 

        print(f"Detailed shots data saved to {output_path}")
        return shot_event_df

    def create_shot_tracking_df(self, output_filename="shot_tracking_df.csv"):
        """
        Creates a shot tracking dataframe from the shot event dataframe and the tracking dataframe.
        """
        shot_tracking_list = []
        json_files = [f for f in os.listdir(self.path_events) if f.endswith('.json') and not f.startswith('.DS_Store')]
        
        for k, json_file in enumerate(json_files):
            game_id = os.path.splitext(json_file)[0]
            print(f"Processing game {k}: {game_id}")
            
            try:
                tracking_df = pd.read_parquet(self.path_tracking / f'{game_id}_tracking.parquet')
                events_df = self._load_single_event_json(json_file)
            except Exception as e:
                print(f"  Error loading data for game {game_id}: {e}")
                continue

            events_df['event_description'] = events_df['event_type_id'].map(self.dict_event_names)
            events_df['game_id'] = int(game_id)
            tracking_df['game_id'] = int(game_id)

            shot_event_df = events_df[events_df['event_type_id'].isin([13, 14, 15, 16])].copy()
            no_shots_initial = len(shot_event_df)
            if shot_event_df.empty:
                print(f"  No shots found in events for game {game_id}")
                continue

            shot_event_df.loc[:, 't_frame_id'] = (shot_event_df['game_id'].astype(str) + '_' + 
                                            shot_event_df['current_phase'].astype(str) + '_' + 
                                            shot_event_df['timeelapsed'].astype(str))
            tracking_df.loc[:, 't_frame_id'] = (tracking_df['game_id'].astype(str) + '_' + 
                                                tracking_df['current_phase'].astype(str) + '_' + 
                                                tracking_df['timeelapsed'].astype(str))

            for index, row in shot_event_df.iterrows():
                if row['event_type_id'] == 16: # Goal
                    try:
                        original_t_frame_id_parts = row['t_frame_id'].split('_')
                        if len(original_t_frame_id_parts) == 3:
                            game_id_parsed_str, current_phase_str, timeelapsed_str = original_t_frame_id_parts
                            game_id_parsed = str(game_id_parsed_str) 
                            current_phase_parsed = int(current_phase_str)
                            timeelapsed_parsed = float(timeelapsed_str)

                            player_id_shot = row['playerId']
                            if pd.isna(player_id_shot): 
                                continue 

                            distance_min = 10_000
                            best_timeelapsed_for_shot = timeelapsed_parsed
                            
                            step_size = 0.4
                            start_time_interval = round(timeelapsed_parsed - step_size * 4, 2)
                            stop_time_interval = round(timeelapsed_parsed + step_size * 1, 2)
                            interval = np.arange(start_time_interval, stop_time_interval + step_size, step_size)

                            # Find the closest tracking frame to the shot event
                            for t_interval in interval:
                                t_interval = round(t_interval, 2)
                                player_frame = tracking_df[
                                    (tracking_df['game_id'] == game_id_parsed) &
                                    (tracking_df['current_phase'] == current_phase_parsed) &
                                    (tracking_df['timeelapsed'] == t_interval) &
                                    (tracking_df['player_id'] == player_id_shot)
                                ]
                                ball_frame = tracking_df[
                                    (tracking_df['game_id'] == game_id_parsed) &
                                    (tracking_df['current_phase'] == current_phase_parsed) &
                                    (tracking_df['timeelapsed'] == t_interval) &
                                    (tracking_df['player_id'] == 'aaaaaaaaaaaaaaaaaaaaaaaaa') # Ball ID
                                ]

                                if not player_frame.empty and not ball_frame.empty:
                                    player_x = player_frame['pos_x'].iloc[0]
                                    player_y = player_frame['pos_y'].iloc[0]
                                    ball_x = ball_frame['pos_x'].iloc[0]
                                    ball_y = ball_frame['pos_y'].iloc[0]
                                    distance = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
                                    if distance < distance_min:
                                        distance_min = distance
                                        best_timeelapsed_for_shot = t_interval
                            shot_event_df.at[index, 't_frame_id'] = f'{game_id_parsed}_{current_phase_parsed}_{best_timeelapsed_for_shot}'
                        else:
                            print(f"  Warning: t_frame_id '{row['t_frame_id']}' for game {game_id} could not be parsed for goal processing.")
                    except Exception as e_goal_proc:
                        print(f"  Error processing goal t_frame_id for game {game_id}, index {index}: {e_goal_proc}")


            merged_df = shot_event_df.merge(tracking_df, on='t_frame_id', how='left', suffixes=('_event', '_track'))
            shot_tracking_list.append(merged_df)
            
            print(f"  Game id: {game_id}, shots: {no_shots_initial}, tracking frames: {len(merged_df)}")
            if not merged_df.empty and no_shots_initial > 0:
                 print(f"  Check - players and ball per shot event: {len(merged_df)/no_shots_initial:.2f}")

        if not shot_tracking_list:
            print("No shot data was processed after iterating all files.")
            return pd.DataFrame()

        shot_tracking_df = pd.concat(shot_tracking_list, ignore_index=True)
        
        # Rename columns and drop unnecessary ones, handling potential missing columns
        rename_map = {
            'game_id_event': 'game_id', 'timeelapsed_event': 'timeelapsed', 
            'current_phase_event': 'current_phase', 'contestantId_event': 'contestantId'
        }
        shot_tracking_df.rename(columns={k: v for k, v in rename_map.items() if k in shot_tracking_df.columns}, inplace=True)
        cols_to_drop = ['current_phase_track', 'timeelapsed_track', 'game_id_track', 'contestantId_track', 'team_id_shooter'] # team_id_shooter from original
        shot_tracking_df.drop(columns=[col for col in cols_to_drop if col in shot_tracking_df.columns], inplace=True)
        
        if not shot_tracking_df.empty:
             shot_tracking_df.loc[:, 't_frame_id'] = (shot_tracking_df['game_id'].astype(str) + '_' + 
                                                     shot_tracking_df['current_phase'].astype(str) + '_' + 
                                                     shot_tracking_df['timeelapsed'].astype(str))

        output_path = self.processed_data_dir / output_filename
        shot_tracking_df.to_csv(output_path, index=False)
        print(f"Tracking augmented shots data saved to {output_path}")
        return shot_tracking_df

    def _find_goalkeeper(self, tracking_df: pd.DataFrame):
        """
        Finds the goalkeepers for a given tracking dataframe.
        """
        df_first_frame = tracking_df[(tracking_df['timeelapsed'] == 0.00) & (tracking_df['current_phase'] == 1)]
        if df_first_frame.empty: return pd.DataFrame(columns=['team_id', 'player_id', 'jersey_no'])

        left_team = df_first_frame[df_first_frame['dop'] == "L"]
        right_team = df_first_frame[df_first_frame['dop'] == "R"]

        goalkeepers_list = []
        if not left_team.empty:
            left_goalkeeper = left_team[left_team['pos_x'] == left_team['pos_x'].min()]
            goalkeepers_list.append(left_goalkeeper)
        if not right_team.empty:
            right_goalkeeper = right_team[right_team['pos_x'] == right_team['pos_x'].max()]
            goalkeepers_list.append(right_goalkeeper)
        
        if not goalkeepers_list: return pd.DataFrame(columns=['team_id', 'player_id', 'jersey_no'])
        
        goalkeepers_df = pd.concat(goalkeepers_list)
        return goalkeepers_df[['team_id', 'player_id', 'jersey_no']]

    def _create_goalkeepers_df(self, output_filename="goalkeepers.csv"):
        """
        Creates a goalkeeper dataframe from the tracking dataframe.
        """
        all_goalkeepers_list = []
        tracking_files = [f for f in os.listdir(self.path_tracking) if f.endswith('_tracking.parquet')]

        for file_name in tracking_files:
            game_id_str = file_name.replace('_tracking.parquet', '')
            try:
                game_id = int(game_id_str)
            except ValueError:
                print(f"Could not parse game_id from {file_name}")
                game_id = game_id_str # Keep as string if parsing fails

            try:
                tracking_df = pd.read_parquet(
                    self.path_tracking / file_name,
                    filters=[[('timeelapsed', '==', 0.0), ('current_phase', '==', 1)]]
                )
            except Exception as e:
                print(f"Error reading or filtering parquet file {file_name}: {e}")
                continue 

            if tracking_df.empty:
                print(f"No data for t=0.0, phase=1 in {file_name}. GK search skipped for this game.")
                # Create an empty df with correct columns to ensure concat works smoothly
                goalkeepers_for_game = pd.DataFrame(columns=['team_id', 'player_id', 'jersey_no'])
            else:
                goalkeepers_for_game = self._find_goalkeeper(tracking_df) # Corrected method call

            goalkeepers_for_game['game_id'] = game_id # Add game_id
            all_goalkeepers_list.append(goalkeepers_for_game)

        if not all_goalkeepers_list:
            print("No goalkeeper data generated after processing all files.")
            # Return an empty DataFrame with expected columns
            return pd.DataFrame(columns=['team_id', 'player_id', 'jersey_no', 'game_id'])

        final_goalkeepers_df = pd.concat(all_goalkeepers_list, ignore_index=True)

        # Ensure 'game_id' column is present even if all sub-dataframes were empty before concat
        # (though adding 'game_id' to each sub-df should prevent this)
        if 'game_id' not in final_goalkeepers_df.columns and not final_goalkeepers_df.empty:
             print("Warning: 'game_id' column was missing after concat.")
        elif final_goalkeepers_df.empty and 'game_id' not in final_goalkeepers_df.columns :
             final_goalkeepers_df = pd.DataFrame(columns=['team_id', 'player_id', 'jersey_no', 'game_id'])

        output_path = self.processed_data_dir / output_filename
        try:
            final_goalkeepers_df.to_csv(output_path, index=False)
            print(f"Goalkeepers data saved to {output_path}")
        except Exception as e:
            print(f"Error saving goalkeepers data to {output_path}: {e}")
        return final_goalkeepers_df

    def load_goalkeepers_df(self, filename="goalkeepers.csv"):
        return pd.read_csv(self.processed_data_dir / filename)
    
    def _enrich_tracking_data(self, tracking_df: pd.DataFrame, 
                              event_df: pd.DataFrame, 
                              goalkeepers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrichs the tracking data with shooter, goalkeeper, and team status.
        """
        print("Enriching tracking data with shooter, goalkeeper, and team status...")
        enriched_tracking_df = tracking_df.copy()

        shooter_player_map = event_df.set_index('t_frame_id')['player_id']
        enriched_tracking_df['shooter_playerId_event'] = enriched_tracking_df['t_frame_id'].map(shooter_player_map)
        enriched_tracking_df['shooter'] = enriched_tracking_df['player_id'] == enriched_tracking_df['shooter_playerId_event']

        if not goalkeepers_df.empty and 'player_id' in goalkeepers_df.columns and 'game_id' in goalkeepers_df.columns:
            goalkeepers_df['is_gk_marker'] = True
            enriched_tracking_df = pd.merge(enriched_tracking_df, 
                                   goalkeepers_df[['game_id', 'player_id', 'is_gk_marker']], 
                                   on=['game_id', 'player_id'], 
                                   how='left')
            enriched_tracking_df['Goalkeeper'] = np.where(enriched_tracking_df['is_gk_marker'] == True, 'Yes', 'No')
            enriched_tracking_df.drop(columns=['is_gk_marker'], inplace=True)
        else:
            print("Warning: Goalkeepers data is empty or missing required columns. 'Goalkeeper' status will be 'No' for all.")
            enriched_tracking_df['Goalkeeper'] = 'No'

        # Get team_id for each shooter by frame
        shooter_teams = enriched_tracking_df[enriched_tracking_df['shooter'] == True][['t_frame_id', 'team_id']]
        shooter_team_map = shooter_teams.set_index('t_frame_id')['team_id']
        
        # Map shooter team_id to all rows with same frame
        enriched_tracking_df['shooter_team_id'] = enriched_tracking_df['t_frame_id'].map(shooter_team_map)
        
        # Set team status based on if team_id matches shooter's team
        enriched_tracking_df['team_status'] = np.where(
            enriched_tracking_df['team_id'] == enriched_tracking_df['shooter_team_id'],
            'attacking',
            'defending'
        )
        
        # Clean up temporary columns used for mapping
        enriched_tracking_df.drop(columns=['shooter_playerId_event', 'shooter_teamId_event'], inplace=True, errors='ignore')
        return enriched_tracking_df
    
    def _filter_data_for_gk_and_shooter(self, event_df: pd.DataFrame, 
                                           tracking_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters the data for shots usable for advanced features with a goalkeeper and shooter.
        """
        print("Filtering data for shots usable for advanced features...")
        tracking_frame_ids = tracking_df['t_frame_id'].unique()
        filtered_shots = event_df[event_df['t_frame_id'].isin(tracking_frame_ids)].copy()

        shooter_in_frames = tracking_df[tracking_df['shooter'] == True]['t_frame_id'].unique()
        filtered_shots = filtered_shots[filtered_shots['t_frame_id'].isin(shooter_in_frames)]

        gk_in_frames = tracking_df[
            (tracking_df['Goalkeeper'] == 'Yes') & 
            (tracking_df['team_status'] == 'defending')
        ]['t_frame_id'].unique()
        filtered_shots = filtered_shots[filtered_shots['t_frame_id'].isin(gk_in_frames)]
        
        valid_tracking_frames_df = tracking_df[
            tracking_df['t_frame_id'].isin(filtered_shots['t_frame_id'])
        ].copy()
        
        print(f"Filtered down to {len(filtered_shots)} shots from {len(event_df)} initial detailed shots.")
        return filtered_shots, valid_tracking_frames_df

    def _normalize_tracking_coordinates(self, tracking_df: pd.DataFrame, 
                                     event_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the tracking coordinates to a consistent direction of play for attacking team.
        """
        print("Normalizing tracking coordinates to a consistent direction of play for attacking team...")
        normalized_df = tracking_df.copy()
        
        def switch_playing_direction(group):
            # Iterate through each row in the group
            for index, row in group.iterrows():
                if row['dop'] == 'R' and row['team_status'] == 'attacking':
                    group['pos_x'] = -group['pos_x']
                    group['pos_y'] = -group['pos_y'] 
                    group['speed_x'] = -group['speed_x']
                    group['speed_y'] = -group['speed_y']
                    return group  # Return the group after the first match
            return group
            
        normalized_df = normalized_df.groupby('t_frame_id').apply(switch_playing_direction).reset_index(drop=True)
        return normalized_df