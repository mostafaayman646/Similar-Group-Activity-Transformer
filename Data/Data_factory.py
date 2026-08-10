from Data.Dataset_enum import Dataset
from Data.Providers import FIFAWC22


def build_sequences(dataset: Dataset, data_dir_path: str, json_id: str) -> dict:
    """
    Returns:
        {
            match_id: {
                seq_id: {
                    frame_num: {
                        'player_position': ...,
                        'ball_position': ...,
                        'time': ...,
                        'event_type': {'game_event_type': int, 'home_ball': bool, 'possession_event_type': int},
                    },
                    ...
                    'total_num_frames': <int>,
                }
            }
        }

    """
    if dataset == Dataset.FIFA_WC_2022:
        pipeline = FIFAWC22(data_dir_path, json_id)
    else:
        raise NotImplementedError(f"No adapter registered for dataset: {dataset}")

    match_id = pipeline.get_match_id()
    sequences = pipeline.build_sequences()

    return {match_id: sequences}