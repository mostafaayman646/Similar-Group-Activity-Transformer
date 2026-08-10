import bz2
import json
import os

from Data.Data_interface import Data_Pipeline


class FIFAWC22(Data_Pipeline):
    def __init__(self, data_dir_path: str, json_id: str):
        self.load_data(data_dir_path, json_id)
        self._filter_frames()
        self.store_stadium_dimensions()
        self.build_jersey_maps()

    def load_data(self, data_dir_path: str, json_id: str):
        """
        Loads tracking data, metadata, and roster data.
        """
        tracking_path = os.path.join(data_dir_path,f'Tracking Data/{json_id}.jsonl.bz2')
        meta_path = os.path.join(data_dir_path,f'Metadata/{json_id}.json')
        roster_path = os.path.join(data_dir_path,f'Rosters/{json_id}.json')
        
        # 1. Load tracking data
        frames = []
        with bz2.open(tracking_path, 'rt') as f:
            for line in f:
                if line.strip():
                    frames.append(json.loads(line))

        frames.sort(key=lambda frame: frame['videoTimeMs'])
        self.frames = {frame['frameNum']: frame for frame in frames}

        # 2. Load metadata
        self.meta = {}
        with open(meta_path, 'rt') as f:
            self.meta = json.load(f)[0]

        # 3. Load roster data
        self.roster = []
        with open(roster_path, 'rt') as f:
            self.roster = json.load(f)

    def store_stadium_dimensions(self):
        pitch = self.meta['stadium']['pitches'][0]
        self.pitch_length = pitch['length']
        self.pitch_width = pitch['width']

    def build_jersey_maps(self):
        home_team_id = self.meta['homeTeam']['id']
        away_team_id = self.meta['awayTeam']['id']

        self.home_jersey_to_id = {}
        self.away_jersey_to_id = {}

        for entry in self.roster:
            team_id = entry['team']['id']
            shirt_num = entry['shirtNumber']
            player_id = entry['player']['id']

            if team_id == home_team_id:
                self.home_jersey_to_id[shirt_num] = player_id
            elif team_id == away_team_id:
                self.away_jersey_to_id[shirt_num] = player_id

    def _normalize_coordinates(self, x: float, y: float) -> tuple:
        norm_x = (2 * x) / self.pitch_length
        norm_y = (2 * y) / self.pitch_width
        return norm_x, norm_y

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

    def _format_players(self, players: list, jersey_to_id: dict) -> list:
        formatted = []
        for p in players or []:
            shirt_num = str(p['jerseyNum'])
            x, y = self._normalize_coordinates(p['x'], p['y'])
            formatted.append({
                'id': jersey_to_id[shirt_num],
                'x': x,
                'y': y
            })
        return formatted

    def get_players_position(self) -> dict:
        return {
            frame_num: {
                'home': self._format_players(
                    frame.get('homePlayersSmoothed'),
                    self.home_jersey_to_id
                ),
                'away': self._format_players(
                    frame.get('awayPlayersSmoothed'),
                    self.away_jersey_to_id
                )
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