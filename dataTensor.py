import torch
import bz2
import json
import pandas as pd
import os

class FIFAWC22:
    def __init__(self, folder_path, game_id,save_tensor = False):
        self.folder_path = folder_path
        self.game_id = game_id
        
        self.load_event_data()
        self.load_tracking_data()
        self.jersey_to_id_mapping()
        self.extract_sequence_tracking()        
        
        if save_tensor:
            self.save_sequences()
    
    def load_event_data(self):
        with open(f'{self.folder_path}/Event Data/{self.game_id}.json', 'rt') as f:
            events_data = json.load(f)
        self.events_df = pd.DataFrame(events_data)
        
        #Remove 0 duration and nan sequences
        self.events_df.drop(self.events_df[(self.events_df['duration'] == 0) & self.events_df['sequence'].isna()].index, inplace=True)
        
        # Group by sequence to find the start of the first event and end of the last event
        self.sequences = self.events_df.groupby('sequence').agg(
            start_time=('startTime', 'min'),
            end_time=('endTime', 'max')
        ).reset_index()
        
        print(f"found: {len(self.sequences)} sequences")
    
    def load_tracking_data(self):
        self.tracking_frames = []
        with bz2.open(f'{self.folder_path}/Tracking Data/{self.game_id}.jsonl.bz2', 'rt') as f:
            for line in f:
                if line.strip():
                    self.tracking_frames.append(json.loads(line))
        self.tracking_frames = sorted(self.tracking_frames, key=lambda x: x['videoTimeMs'])
    
    def jersey_to_id_mapping(self):
        self.home_jersey_to_id = {}
        self.away_jersey_to_id = {}
        
        for _, event_row in self.events_df.iterrows():
            for p in event_row.get('homePlayers', []):
                j_num = str(p.get('jerseyNum'))
                p_id = p.get('playerId')
                if j_num != 'None' and p_id is not None:
                    self.home_jersey_to_id[j_num] = p_id
            
            for p in event_row.get('awayPlayers', []):
                j_num = str(p.get('jerseyNum'))
                p_id = p.get('playerId')
                if j_num != 'None' and p_id is not None:
                    self.away_jersey_to_id[j_num] = p_id
    
    def extract_sequence_tracking(self):
        self.extracted_sequences = []
        
        frame_idx = 0
        num_frames = len(self.tracking_frames)
        
        for _, row in self.sequences.iterrows():
            seq_id = row['sequence']
            
            start_ms = row['start_time'] * 1000
            end_ms = row['end_time'] * 1000
            
            # Fast-forward pointer
            while frame_idx < num_frames and self.tracking_frames[frame_idx]['videoTimeMs'] < start_ms:
                frame_idx += 1
                
            seq_frames = []
            curr_idx = frame_idx
            
            while curr_idx < num_frames and self.tracking_frames[curr_idx]['videoTimeMs'] <= end_ms:
                frame = self.tracking_frames[curr_idx]
                
                # Pre-allocate a (23, 4) tensor with zeros.
                frame_tensor = torch.zeros((23, 4), dtype=torch.float32)
                
                # 1. Home Players -> Fixed indices 0 to 10 | Role = 0
                for i, p in enumerate(frame['homePlayers'][:11]):
                    frame_tensor[i, 0] = self.home_jersey_to_id.get(str(p.get('jerseyNum')), -1)
                    frame_tensor[i, 1] = p['x'] / 52.5  # Normalize x to [0, 1]
                    frame_tensor[i, 2] = p['y'] / 34.0
                    # Role is already 0.0
                
                # 2. Away Players -> Fixed indices 11 to 21 | Role = 1
                for i, p in enumerate(frame['awayPlayers'][:11]):
                    frame_tensor[11 + i, 0] = self.away_jersey_to_id.get(str(p.get('jerseyNum')), -1)
                    frame_tensor[11 + i, 1] = p['x'] / 52.5
                    frame_tensor[11 + i, 2] = p['y'] / 34.0
                    frame_tensor[11 + i, 3] = 1.0  # Set Role to 1
                
                # 3. Ball -> Fixed index 22 | Role = 2
                balls = frame['balls']
                if balls:
                    b = balls[0]
                    # ID is already 0.0 
                    frame_tensor[22, 1] = b['x'] / 52.5
                    frame_tensor[22, 2] = b['y'] / 34.0
                    frame_tensor[22, 3] = 2.0  # Set Role to 2
                
                seq_frames.append(frame_tensor)
                curr_idx += 1
                
            # Stack the list of (23, 4) tensors into a single (T, 23, 4) tensor
            if seq_frames:
                sequence_tensor = torch.stack(seq_frames)
                self.extracted_sequences.append({
                    'sequence': seq_id,
                    'tracking_data': sequence_tensor
                })

    def save_sequences(self):
        save_dir = 'DataBase'  
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, f"{self.game_id}.pt")
        if os.path.exists(file_path):
            print(f"Data for game {self.game_id} already saved. Skipping...")
            return

        # self.extracted_sequences is a list of dicts.
        # PyTorch can save standard Python lists perfectly!
        torch.save(self.extracted_sequences, file_path)
        print(f"Saved {len(self.extracted_sequences)} sequences to {file_path}")

if '__main__' == __name__:

    game_ids = [
        '10511', '3812', '3813', '3814', '3815', '3816', '3817', '3818',
        '3819', '3820', '3821', '3822', '3823', '3824', '3825', '3826',
        '3827', '3828', '3829', '3830', '3831', '10502', '10503', '10504',
        '10505', '10507', '10509', '10512', '10513', '10514', '10515',
        '10516', '3834', '3835', '3836', '3837', '3838', '3839', '3840',
        '3841', '3842', '3843', '3844', '3845', '3846', '3847', '3857',
        '3858', '3859', '3848', '3849', '3850', '3852', '3832', '3853',
        '3854', '3855', '3856', '3851', '3833', '10508', '10506', '10517',
        '10510'
    ]

    for gid in game_ids:
        print(f"Processing Game {gid}...")
        # This will load, process, and save the .pt file for the game
        data = FIFAWC22('FIFA World Cup 2022', gid, save_tensor=True)