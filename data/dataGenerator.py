import bz2
import json
import pandas as pd
import numpy as np
import torch
import os

class FIFAWC22:
    def __init__(self, folder_path, game_id, sample_size=100, pre_buffer=10, post_buffer=3):
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

                raw_anchors.append({
                    'sequence_id': seq_id,
                    'event_time': row.get('eventTime'),
                    'single_label': single_label,
                    'is_home_possession': is_home_possession,
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
                'is_home_possession': events[0]['is_home_possession']
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
            else:
                supcon_label = None
                is_home_possession = None
            
            # --- BALL EXTRACTION (Role = 2, Player_ID = 0) ---
            balls = row.get('ballsSmoothed')
            if not isinstance(balls, list) or len(balls) == 0:
                balls = row.get('balls', [])
                
            ball_x, ball_y = np.nan, np.nan
            if isinstance(balls, list) and len(balls) > 0:
                ball_x = balls[0].get('x', np.nan)
                ball_y = balls[0].get('y', np.nan)
                
            extracted_data.append({
                'seq_id': seq_id,
                'videoTimeMs': video_time,
                'role': 2,        # 2 = Ball
                'player_id': 0,   # Ball ID is 0
                'x': ball_x,
                'y': ball_y,
                'rel_x': 0.0,
                'rel_y': 0.0,
                'supcon_label': supcon_label,           # NEW: Appending SupCon Label
                'is_home_possession': is_home_possession # NEW: Appending Possession Flag
            })
            
            # --- PLAYER EXTRACTION HELPER ---
            def process_players(players_list, role_int, jersey_map):
                if not isinstance(players_list, list):
                    return
                    
                for p in players_list:
                    p_x = p.get('x', np.nan)
                    p_y = p.get('y', np.nan)
                    
                    # Safely extract jerseyNum and map to playerId
                    j_num = str(p.get('jerseyNum')) if p.get('jerseyNum') is not None else None
                    p_id = jersey_map.get(j_num, None)
                    
                    # Calculate relative coordinates
                    if pd.notna(ball_x) and pd.notna(p_x):
                        rel_x = p_x - ball_x
                        rel_y = p_y - ball_y
                    else:
                        rel_x, rel_y = np.nan, np.nan
                        
                    extracted_data.append({
                        'seq_id': seq_id,
                        'videoTimeMs': video_time,
                        'role': role_int,
                        'player_id': p_id,
                        'x': p_x,
                        'y': p_y,
                        'rel_x': rel_x,
                        'rel_y': rel_y,
                        'supcon_label': supcon_label,           # NEW: Appending SupCon Label
                        'is_home_possession': is_home_possession # NEW: Appending Possession Flag
                    })
                    
            # Extract Home Players (Role = 0)
            home_players = row.get('homePlayersSmoothed')
            if not isinstance(home_players, list) or len(home_players) == 0:
                home_players = row.get('homePlayers', [])
            process_players(home_players, 0, self.home_jersey_map)
            
            # Extract Away Players (Role = 1)
            away_players = row.get('awayPlayersSmoothed')
            if not isinstance(away_players, list) or len(away_players) == 0:
                away_players = row.get('awayPlayers', [])
            process_players(away_players, 1, self.away_jersey_map)
            
        # Compile Final DataFrame
        self.final_extracted_df = pd.DataFrame(extracted_data)
        print(f"Extraction complete. Created tabular mapping with {len(self.final_extracted_df)} records.")
        print(f"Sequence 10 now has: {len(self.final_extracted_df[self.final_extracted_df['seq_id'] == 10])} frames")
        print(f"Shape: {self.final_extracted_df.shape}")


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