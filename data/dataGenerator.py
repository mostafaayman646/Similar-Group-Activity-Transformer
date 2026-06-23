import bz2
import json
import math

import pandas as pd
import numpy as np
import torch
import os

class FIFAWC22:
    def __init__(self, folder_path, game_id, sample_size=100, pre_buffer=10, post_buffer=3,save_Tensor = False):
        self.folder_path = folder_path
        self.game_id = game_id
        self.sample_size = sample_size
        self.pre_buffer = pre_buffer
        self.post_buffer = post_buffer
        
        self.load_event_data()
        self.get_important_sequences()
        self.load_tracking_data()
        self.sample_sequence_frames()
        self.extract_per_frame_info()
        self.post_process_ball_data()
        self.validate_extraction(sample_seq=10)
    
    def load_event_data(self):
        with open(f'{self.folder_path}/Event Data/{self.game_id}.json', 'rt') as f:
            events_data = json.load(f)
        self.events_df = pd.DataFrame(events_data)
        
        # Group by sequence
        self.sequences = self.events_df.groupby('sequence').agg(
            start_time=('startTime', 'min'),
            end_time=('endTime', 'max')
        )
        print(f"Found: {len(self.sequences)} valid sequences")
    
    def get_important_sequences(self):
        """
        Identifies tactical anchors, chains compound labels, and calculates 
        precise elastic tracking windows clamped to sequence boundaries.
        """
        target_event_types = {'SH', 'CR', 'FO'} # Updated focus
        seq_boundaries = {}
        raw_anchors = []

        # 1. Map global sequence boundaries to prevent bleeding
        for _, row in self.events_df.iterrows():
            seq_id = row.get('sequence')
            e_time = row.get('eventTime')
            if pd.notna(seq_id) and pd.notna(e_time):
                if seq_id not in seq_boundaries:
                    seq_boundaries[seq_id] = {'start': e_time, 'end': e_time}
                else:
                    seq_boundaries[seq_id]['start'] = min(seq_boundaries[seq_id]['start'], e_time)
                    seq_boundaries[seq_id]['end'] = max(seq_boundaries[seq_id]['end'], e_time)

        # 2. Extract anchors and dynamically build labels
        for _, row in self.events_df.iterrows():
            poss_event = row.get('possessionEvents')
            if not isinstance(poss_event, dict): 
                continue

            event_type = poss_event.get('possessionEventType')
            if event_type in target_event_types:
                seq_id = row.get('sequence')

                sub_type, outcome = None, None
                if event_type == 'SH':
                    sub_type = poss_event.get('shotType')
                    outcome = poss_event.get('shotOutcomeType')
                elif event_type == 'CR':
                    sub_type = poss_event.get('crossType')
                    outcome = poss_event.get('crossOutcomeType')
                elif event_type == 'FO':
                    foul_data = row.get('fouls', {})
                    if isinstance(foul_data, dict):
                        sub_type = foul_data.get('finalOffenseType')
                        outcome = foul_data.get('finalFoulOutcomeType')

                # Build single event label
                label_parts = [event_type]
                if sub_type: label_parts.append(sub_type)
                if outcome: label_parts.append(outcome)
                single_label = "_".join(label_parts)

                # Safely extract possession flag
                game_events = row.get('gameEvents', {})
                is_home_possession = game_events.get('homeTeam') if isinstance(game_events, dict) else None

                # Extract Attacking Direction
                stadium_meta = row.get('stadiumMetadata', {})
                atk_dir = 'R'  # Default to Right
                pitch_length = 105.0  # Safe FIFA default
                pitch_width = 68.0
                if isinstance(stadium_meta, dict):
                    atk_dir = stadium_meta.get('teamAttackingDirection', 'R')
                    pitch_length = stadium_meta.get('pitchLength', 105.0)
                    pitch_width = stadium_meta.get('pitchWidth', 68.0)

                raw_anchors.append({
                    'sequence_id': seq_id,
                    'event_time': row.get('eventTime'),
                    'single_label': single_label,
                    'is_home_possession': is_home_possession,
                    'attacking_direction': atk_dir,
                    'pitch_length': pitch_length,
                    'pitch_width': pitch_width,
                    'seq_start': seq_boundaries[seq_id]['start'],
                    'seq_end': seq_boundaries[seq_id]['end']
                })

        # 3. Chronological Chaining & Elastic Windows
        sequences = {}
        for anchor in raw_anchors:
            seq_id = anchor['sequence_id']
            if seq_id not in sequences: 
                sequences[seq_id] = []
            sequences[seq_id].append(anchor)

        important_seqs_data = []
        for seq_id, events in sequences.items():
            events.sort(key=lambda x: x['event_time'])

            compound_label = " __ ".join([e['single_label'] for e in events])
            first_time = events[0]['event_time']
            last_time = events[-1]['event_time']

            # Stretch window and clamp to boundaries
            start_t = max(first_time - self.pre_buffer, events[0]['seq_start'])
            end_t = min(last_time + self.post_buffer, events[0]['seq_end'])

            important_seqs_data.append({
                'sequence_id': seq_id,
                'start_time': start_t,
                'end_time': end_t,
                'supcon_label': compound_label,
                'is_home_possession': events[0]['is_home_possession'],
                'attacking_direction':events[0]['attacking_direction'],
                'pitch_length':events[0]['pitch_length'],
                'pitch_width':events[0]['pitch_width'],
            })

        # Overwrite DataFrame with new focused intervals. 
        # load_tracking_data() will automatically use these new start/end times.
        self.important_sequence_times = pd.DataFrame(important_seqs_data).set_index('sequence_id')
        print(f"Important sequences (Elastic Windows): {len(self.important_sequence_times)}")
    
    def load_tracking_data(self):
        print("Loading and filtering tracking data...")
        self.tracking_frames = []
        file_path = f'{self.folder_path}/Tracking Data/{self.game_id}.jsonl.bz2'
        
        # Pre-extract to a native Python list for much faster lookup inside the loop
        intervals = [
            (row.Index, row.start_time * 1000, row.end_time * 1000)
            for row in self.important_sequence_times.itertuples()
        ]
        
        with bz2.open(file_path, 'rt') as f:
            for line in f:
                if line.strip():
                    frame = json.loads(line)
                    t = frame['videoTimeMs']
                    seq_id = -1
                    
                    # Iterate over the pre-calculated list, not the pandas dataframe
                    for seq, start_ms, end_ms in intervals:
                        if start_ms <= t <= end_ms:
                            seq_id = int(seq)
                            break
                    
                    if seq_id != -1:
                        frame['seq_id'] = seq_id
                        self.tracking_frames.append(frame)
                
        # Sort tracking frames by video time to maintain chronological order
        self.tracking_frames = sorted(self.tracking_frames, key=lambda x: x['videoTimeMs'])
        self.tracking_df = pd.DataFrame(self.tracking_frames)
        print(f"Loaded {len(self.tracking_df)} total tracking frames.")
        print(len(self.tracking_df[self.tracking_df['seq_id'] == 10]))
    
    def sample_sequence_frames(self):
        """
        Calculates the target number of frames per unique event based on self.sample_size.
        Reallocates unused frame quotas from short events to longer events to ensure 
        the final sequence hits the target sample_size.
        """
        print("Downsampling tracking frames per sequence event...")
        sampled_frames_list = []
        
        for seq_id, seq_df in self.tracking_df.groupby('seq_id'):
            seq_df = seq_df.copy()
            
            # 1. Fill NaNs so we don't lose the continuous tracking data
            seq_df['game_event_id'] = seq_df['game_event_id'].ffill().bfill()
            
            unique_events = seq_df['game_event_id'].unique()
            num_events = len(unique_events)
            
            if num_events == 0:
                continue
            
            # Initial target per event
            base_target = self.sample_size // num_events
            remainder = self.sample_size % num_events
            
            targets = {}
            for i, ev_id in enumerate(unique_events):
                targets[ev_id] = base_target + (1 if i < remainder else 0)
            
            # 2. First pass: Check for "short" events and calculate the deficit
            deficit = 0
            for ev_id in unique_events:
                actual_frames = len(seq_df[seq_df['game_event_id'] == ev_id])
                if actual_frames < targets[ev_id]:
                    # We have fewer frames than the target. Calculate how many we are short.
                    deficit += (targets[ev_id] - actual_frames)
                    targets[ev_id] = actual_frames # Cap the target to what's available
            
            # 3. Reallocate the deficit to events that have extra frames
            if deficit > 0:
                for ev_id in unique_events:
                    actual_frames = len(seq_df[seq_df['game_event_id'] == ev_id])
                    extra_capacity = actual_frames - targets[ev_id]
                    
                    if extra_capacity > 0:
                        # Give this event as much of the deficit as it can handle
                        added_frames = min(extra_capacity, deficit)
                        targets[ev_id] += added_frames
                        deficit -= added_frames
                        
                    if deficit == 0:
                        break # All missing frames reallocated!

            # 4. Final Pass: Sample the frames using the updated targets
            for ev_id in unique_events:
                event_frames = seq_df[seq_df['game_event_id'] == ev_id]
                t_frames = targets[ev_id]
                a_frames = len(event_frames)
                
                if a_frames > t_frames:
                    indices = np.linspace(0, a_frames - 1, t_frames, dtype=int)
                    sampled_frames_list.append(event_frames.iloc[indices])
                else:
                    sampled_frames_list.append(event_frames)
                    
        # Reconstruct the DataFrame and sort chronologically
        if sampled_frames_list:
            self.sampled_tracking_df = pd.concat(sampled_frames_list).sort_values(by=['seq_id', 'videoTimeMs']).reset_index(drop=True)
            print(f"Downsampling complete. New total frames: {len(self.sampled_tracking_df)}")
            print(f"Sequence 10 now has: {len(self.sampled_tracking_df[self.sampled_tracking_df['seq_id'] == 10])} frames")
        else:
            self.sampled_tracking_df = pd.DataFrame()
            print("Warning: No frames were left after downsampling.")
    
    def _build_jersey_mappings(self):
        """
        Builds jerseyNum -> playerId mapping for all players (including subs)
        without the heavy overhead of .iterrows().
        """
        print("Building Jersey-to-PlayerID mapping from Event Data...")
        self.home_jersey_map = {}
        self.away_jersey_map = {}
        
        # Fast iteration directly over the Series (ignores NaNs)
        for players_list in self.events_df['homePlayers'].dropna():
            for p in players_list:
                if 'jerseyNum' in p and 'playerId' in p:
                    self.home_jersey_map[str(p['jerseyNum'])] = p['playerId']
                    
        for players_list in self.events_df['awayPlayers'].dropna():
            for p in players_list:
                if 'jerseyNum' in p and 'playerId' in p:
                    self.away_jersey_map[str(p['jerseyNum'])] = p['playerId']
    
    def extract_per_frame_info(self):
        # 1. Ensure mappings exist before extracting
        self._build_jersey_mappings()
        
        print("Extracting per-frame player and ball info...")
        extracted_data = []
        
        # 2. Iterate over the sampled tracking frames per sequence
        for _, row in self.sampled_tracking_df.iterrows():
            seq_id = row.get('seq_id')
            video_time = row.get('videoTimeMs')
            
            # --- Fetch Metadata for this Sequence ---
            # Safely retrieve the compound label and possession flag created in get_important_sequences
            if seq_id in self.important_sequence_times.index:
                seq_metadata = self.important_sequence_times.loc[seq_id]
                supcon_label = seq_metadata['supcon_label']
                is_home_possession = seq_metadata['is_home_possession']
                atk_dir = seq_metadata.get('attacking_direction', 'R')
                p_length = seq_metadata.get('pitch_length', 105.0)
                p_width = seq_metadata.get('pitch_width', 68.0)
            else:
                supcon_label, is_home_possession, atk_dir = None, None, 'R'
                p_length, p_width = 105.0, 68.0

            # --- DIRECTION NORMALIZATION (ATTACKING RIGHT) ---
            # If the team is attacking Left ('L'), direction_mult = -1.0 to flip the pitch.
            direction_mult = -1.0 if atk_dir == 'L' else 1.0
            x_scale = p_length / 2.0
            y_scale = p_width / 2.0

            smooth_ball = row.get('ballsSmoothed', {})
            raw_ball_list = row.get('balls', [])

            ball_x, ball_y, ball_z = np.nan, np.nan, 0.0
            ball_vis, ball_speed = 0.0, 0.0

            # Handle the dictionary correctly
            if isinstance(smooth_ball, dict) and 'x' in smooth_ball:
                bx_val = smooth_ball.get('x')
                by_val = smooth_ball.get('y')
                bz_val = smooth_ball.get('z')

                # Safely apply Math ONLY if the JSON didn't return null
                if bx_val is not None and by_val is not None:
                    # 1. Orient the pitch
                    raw_x = bx_val * direction_mult
                    raw_y = by_val * direction_mult

                    # 2. Normalize to [-1.0, 1.0]
                    ball_x = raw_x / x_scale
                    ball_y = raw_y / y_scale

                ball_z = bz_val if bz_val is not None else 0.0 # Todo Normalize Z


                # Safely pull from raw list if available
                if isinstance(raw_ball_list, list) and len(raw_ball_list) > 0:
                    ball_vis = 1.0 if raw_ball_list[0].get('visibility') == 'VISIBLE' else 0.0
                    ball_speed = raw_ball_list[0].get('speed')
                    ball_speed = ball_speed if ball_speed is not None else 0.0 # Todo ball has no speed

            dist_to_goal = math.hypot(1.0 - ball_x, 0.0 - ball_y) if pd.notna(ball_x) else np.nan

            extracted_data.append({
                'seq_id': seq_id,
                'videoTimeMs': video_time,
                'role': 2,        # 2 = Ball
                'player_id': 0,   # Ball ID is 0
                'x': ball_x,
                'y': ball_y,
                'z': ball_z,
                'speed': ball_speed,
                'visibility': ball_vis,
                'is_attacking': 0.0,
                'dist_to_goal': dist_to_goal,
                # 'rel_x': 0.0,
                # 'rel_y': 0.0,
                'supcon_label': supcon_label,           # NEW: Appending SupCon Label
            })
            
            # --- PLAYER EXTRACTION HELPER ---
            def process_players(smooth_list, raw_list, role_int, jersey_map, is_attacking_team):
                if not isinstance(smooth_list, list):
                    return

                raw_dict = {str(p.get('jerseyNum')): p for p in raw_list} if isinstance(raw_list, list) else {}
                for p in smooth_list:

                    j_num = str(p.get('jerseyNum'))
                    raw_p = raw_dict.get(j_num, {})  # Get the corresponding raw data

                    px_val = p.get('x')
                    py_val = p.get('y')

                    p_x, p_y = np.nan, np.nan

                    if px_val is not None and py_val is not None:
                        p_x = (px_val * direction_mult) / x_scale
                        p_y = (py_val * direction_mult) / y_scale

                    raw_speed = raw_p.get('speed')
                    speed = raw_speed if raw_speed is not None else 0.0

                    visibility = 1.0 if raw_p.get('visibility') == 'VISIBLE' else 0.0
                    is_attacking = 1.0 if is_attacking_team else 0.0
                    dist = math.hypot(1.0 - p_x, 0.0 - p_y) if pd.notna(p_x) else np.nan


                    p_id = jersey_map.get(j_num, None)
                    
                    # Calculate relative coordinates
                    # if pd.notna(ball_x) and pd.notna(p_x):
                    #     rel_x = p_x - ball_x
                    #     rel_y = p_y - ball_y
                    # else:
                    #     rel_x, rel_y = np.nan, np.nan
                        
                    extracted_data.append({
                        'seq_id': seq_id,
                        'videoTimeMs': video_time,
                        'role': role_int,
                        'player_id': p_id,
                        'x': p_x,
                        'y': p_y,
                        'z': 0.0,  # NEW: Pad players with Z=0
                        'speed': speed,  # NEW
                        'visibility': visibility,  # NEW
                        'is_attacking': is_attacking,  # NEW
                        'dist_to_goal': dist,  # NEW
                        # 'rel_x': rel_x,
                        # 'rel_y': rel_y,
                        'supcon_label': supcon_label,           # NEW: Appending SupCon Label

                    })

            # Extract Home Players (Role = 0)
            s_home = row.get('homePlayersSmoothed', [])
            r_home = row.get('homePlayers', [])
            is_home_atk = (is_home_possession == True)
            process_players(s_home, r_home, 0, self.home_jersey_map, is_home_atk)

            # Extract Away Players (Role = 1)
            s_away = row.get('awayPlayersSmoothed', [])
            r_away = row.get('awayPlayers', [])
            is_away_atk = (is_home_possession == False)
            process_players(s_away, r_away, 1, self.away_jersey_map, is_away_atk)

        # Compile Final DataFrame
        self.final_extracted_df = pd.DataFrame(extracted_data)
        print(f"Extraction complete. Created tabular mapping with {len(self.final_extracted_df)} records.")
        print(f"Sequence 10 now has: {len(self.final_extracted_df[self.final_extracted_df['seq_id'] == 10])} frames")
        print(f"Shape: {self.final_extracted_df.shape}")
    
    def post_process_ball_data(self):
        """
        Normalizes ball Z values and calculates ball speed based on frame-to-frame distance.
        To be called after extract_per_frame_info() is completed.
        """
        import numpy as np
        import pandas as pd
        
        # Mask to isolate only the ball records (role == 2)
        ball_mask = self.final_extracted_df['role'] == 2
        
        # Extract ball data and ensure strictly chronological order per sequence
        ball_data = self.final_extracted_df[ball_mask].sort_values(by=['seq_id', 'videoTimeMs'])
        
        # --- 1. Calculate Ball Speed ---
        # Calculate time difference in seconds
        dt_sec = ball_data.groupby('seq_id')['videoTimeMs'].diff() / 1000.0
        
        # Calculate coordinate differences
        dx = ball_data.groupby('seq_id')['x'].diff()
        dy = ball_data.groupby('seq_id')['y'].diff()
        dz = ball_data.groupby('seq_id')['z'].diff()
        
        # Euclidean distance
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Calculate speed (distance / time). Handle division by zero and NaNs for the first frames
        speed = (dist / dt_sec).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        # Update the original DataFrame using the aligned indices
        self.final_extracted_df.loc[ball_data.index, 'speed'] = speed
        
        # --- 2. Normalize Ball Z Values (Standard Scaler) ---
        ball_z = self.final_extracted_df.loc[ball_mask, 'z']
        z_mean = ball_z.mean()
        z_std = ball_z.std()
        
        # Apply standard scaling (z = (x - mean) / std), protecting against division by zero
        if pd.notna(z_std) and z_std > 0:
            self.final_extracted_df.loc[ball_mask, 'z'] = (ball_z - z_mean) / z_std
        else:
            self.final_extracted_df.loc[ball_mask, 'z'] = 0.0
            
        print("Post-processing complete: Ball Z-values standard-scaled and speeds calculated.")
    
    def validate_extraction(self, sample_seq=10):
        """
        Validates the final extracted DataFrame to ensure metadata
        and tracking data merged correctly.
        """
        df = self.final_extracted_df

        print("\n--- Extraction Validation ---")
        print(f"Total Extracted Records: {len(df)}")

        # 1. Validate Columns
        expected_cols = ['seq_id', 'videoTimeMs', 'role', 'player_id', 'x', 'y', 'rel_x', 'rel_y', 'supcon_label',
                         'is_home_possession']
        missing_cols = [col for col in expected_cols if col not in df.columns]

        if missing_cols:
            print(f"WARNING: Missing expected columns: {missing_cols}")
        else:
            print("Successfully verified all required columns are present.")

            # 2. Check for missing metadata (ensures the Elastic Window join worked)
            missing_labels = df['supcon_label'].isna().sum()
            if missing_labels > 0:
                print(f"WARNING: Found {missing_labels} records missing a 'supcon_label'.")
            else:
                print("All records successfully mapped to a SupCon label.")

        # 3. Sequence Specific Validation
        print(f"\n--- Sequence {sample_seq} Validation ---")
        sample_records = df[df['seq_id'] == sample_seq]

        if len(sample_records) > 0:
            print(f"Total records (players + ball): {len(sample_records)}")

            # Group by video time to see how many actual tracking frames exist
            unique_frames = sample_records['videoTimeMs'].nunique()
            print(f"Total unique tracking frames: {unique_frames}")

            # Print the first row as a dictionary to visually verify the data types
            print(f"\nSample Data Row:")
            sample_dict = sample_records.iloc[92].to_dict()
            for key, value in sample_dict.items():
                print(f"  {key}: {value}")
        else:
            print(f"Sequence {sample_seq} not found in extracted data.")
            print("(Note: This is normal if Sequence 10 did not contain a target Shot, Cross, or Foul).")
        print("-----------------------------\n")
    
    # Todo : Function for saving tensor in required format
    # Todo : manual sequence label

if __name__ == '__main__':
    game_ids = [
        # '10511', '3812', '3813', '3814', '3815', '3816', '3817', '3818',
        # '3819', '3820', '3821', '3822', '3823', '3824', '3825', '3826',
        # '3827', '3828', '3829', '3830', '3831', '10502', '10503', '10504',
        # '10505', '10507', '10509', '10512', '10513', '10514', '10515',
        # '10516', '3834', '3835', '3836', '3837', '3838', '3839', '3840',
        # '3841', '3842', '3843', '3844', '3845', '3846', '3847', '3857',
        # '3858', '3859', '3848', '3849', '3850', '3852', '3832', '3853',
        # '3854', '3855', '3856', '3851', '3833', '10508', '10506', '10517',
        '10510'
    ]
    
    for gid in game_ids:
        print(f"Processing Game {gid}...")
        FIFAWC22('FIFA World Cup 2022', gid)