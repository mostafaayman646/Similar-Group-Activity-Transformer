import torch
from torch.utils.data import Dataset
import os
import gc


class FIFASequenceDataset(Dataset):
    def __init__(self, data_dir,match_files, target_frames=230):
        self.target_frames = target_frames
        self.all_tensors = []
        self.all_seq_ids = []

        print("Loading matches into memory.")
        # Find all match files in the DataBase folder
        file_paths = [os.path.join(data_dir, f) for f in match_files]

        # Load each match and append its sequences to our master list
        for path in file_paths:

            if not os.path.exists(path):
                print(f"Warning: {path} not found. Skipping.")
                continue

            match_sequences = torch.load(path,weights_only=False)  # Loads the list of dicts


            for seq_dict in match_sequences:
                self.all_tensors.append(seq_dict['tracking_data'])
                self.all_seq_ids.append(seq_dict['sequence'])

            # Delete the heavy dictionary list from RAM immediately
            del match_sequences

        # Force Python to clean up the deleted dictionaries from memory
        gc.collect()

        print(f"Loaded {len(self.all_tensors)} total plays across {len(file_paths)} matches!")

    def __len__(self):
        return len(self.all_tensors)

    def __getitem__(self, idx):
        # Grab the raw tensor and ID directly from our stripped lists
        raw_tensor = self.all_tensors[idx]  # Shape: [T, 23, 4]
        seq_id = self.all_seq_ids[idx]

        current_frames = raw_tensor.shape[0]

        # Initialize empty arrays for padding
        fixed_coords = torch.zeros((self.target_frames, 23, 2), dtype=torch.float32)
        fixed_roles = torch.zeros((self.target_frames, 23), dtype=torch.int64)

        # Slice the data and cast to the correct Data Types
        actual_coords = raw_tensor[:, :, 1:3].float()
        actual_roles = raw_tensor[:, :, 3].long()

        # Truncate or Pad to hit exactly 100 frames
        if current_frames >= self.target_frames:
            fixed_coords = actual_coords[-self.target_frames:, :, :]
            fixed_roles = actual_roles[-self.target_frames:, :]
        else:
            fixed_coords[-current_frames:, :, :] = actual_coords
            fixed_roles[-current_frames:, :] = actual_roles

        return {
            'coordinates': fixed_coords,#[230,23,2]
            'roles': fixed_roles,#[230,23,1]
            'sequence_id': seq_id #[230,23,1]
        }