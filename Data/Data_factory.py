from Data.Dataset_enum import Dataset
from Data.Providers import FIFAWC22


def build_sequences(dataset: Dataset, data_dir_path: str) -> dict:
    """
    Returns:
        {
            match_id: {
                seq_id: {
                    frame_num: {
                        'player_position': ...,
                        'ball_position': ...,
                        'time': ...,
                        'event_type': {'game_event': ..., 'possession_event': ...},
                    },
                    ...
                    'total_num_frames': <int>,
                }
            }
        }

    """
    if dataset == Dataset.FIFA_WC_2022:
        pipeline = FIFAWC22(data_dir_path)
    else:
        raise NotImplementedError(f"No adapter registered for dataset: {dataset}")

    match_id = pipeline.get_match_id()
    player_positions = pipeline.get_players_position()
    ball_positions = pipeline.get_ball_position()
    event_types = pipeline.get_event_type()
    times = pipeline.get_time()
    sequence_ids = pipeline.get_sequence_id()

    sequences = {}
    
    for frame_num, seq_id in sequence_ids.items():
        sequences.setdefault(seq_id, {})[frame_num] = {
            'player_position': player_positions.get(frame_num),
            'ball_position': ball_positions.get(frame_num),
            'time': times.get(frame_num),
            'event_type': event_types.get(frame_num),
        }

    for seq_frames in sequences.values():
        seq_frames['total_num_frames'] = len(seq_frames)

    return {match_id: sequences}