import bz2
import json

from Data.Data_interface import Data_Pipeline


class FIFAWC22(Data_Pipeline):
    def __init__(self, data_dir_path: str):
        self.load_data(data_dir_path)
        self._filter_frames()

    def load_data(self, data_dir_path: str):
        """
        data_dir_path: path to the game's tracking file,
        e.g. ".../Tracking Data/10510.jsonl.bz2"
        """
        frames = []
        with bz2.open(data_dir_path, 'rt') as f:
            for line in f:
                if line.strip():
                    frames.append(json.loads(line))

        frames.sort(key=lambda frame: frame['videoTimeMs'])
        self.frames = {frame['frameNum']: frame for frame in frames}

    def _filter_frames(self):
        """
        Calculates sequence IDs and drops the frame if:
        - it has no seq_id (nothing happening in it yet), or
        - no player - home or away - was tracked with HIGH confidence
        """
        # 1. Calculate sequence IDs for all frames
        sequence_ids = {}
        current_sequence = None
        for frame_num, frame in self.frames.items():
            game_event = frame.get('game_event')
            if game_event and game_event.get('sequence') is not None:
                current_sequence = game_event['sequence']
            sequence_ids[frame_num] = current_sequence

        # 2. Filter frames based on sequence presence and tracking confidence
        valid_frames = {}
        for frame_num, frame in self.frames.items():
            seq_id = sequence_ids[frame_num]
            
            if seq_id is None:
                continue

            home_players = frame.get('homePlayersSmoothed') or []
            away_players = frame.get('awayPlayersSmoothed') or []
            all_players = home_players + away_players

            if not all_players:
                continue

            if any(player.get('confidence') == 'HIGH' for player in all_players):
                # Store the calculated seq_id directly in the frame for easy retrieval later
                frame['sequence_id'] = seq_id
                valid_frames[frame_num] = frame

        self.frames = valid_frames

    def get_players_position(self) -> dict:
        return {
            frame_num: {
                'home': frame.get('homePlayersSmoothed'),
                'away': frame.get('awayPlayersSmoothed'),
            }
            for frame_num, frame in self.frames.items()
        }

    def get_ball_position(self) -> dict:
        return {
            frame_num: frame.get('balls')
            for frame_num, frame in self.frames.items()
        }

    def get_event_type(self) -> dict:
        return {
            frame_num: {
                'game_event': frame.get('game_event'),
                'possession_event': frame.get('possession_event'),
            }
            for frame_num, frame in self.frames.items()
        }

    def get_time(self) -> dict:
        return {
            frame_num: {
                'videoTimeMs': frame.get('videoTimeMs'),
                'period': frame.get('period'),
                'periodElapsedTime': frame.get('periodElapsedTime'),
                'periodGameClockTime': frame.get('periodGameClockTime'),
            }
            for frame_num, frame in self.frames.items()
        }

    def get_match_id(self):
        first_frame = next(iter(self.frames.values()))
        return first_frame.get('gameRefId')

    def get_sequence_id(self) -> dict:
        return {
            frame_num: frame.get('sequence_id')
            for frame_num, frame in self.frames.items()
        }