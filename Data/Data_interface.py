from abc import ABC, abstractmethod


class Data_Pipeline(ABC):
    @abstractmethod
    def get_players_position(self, frame: dict) -> dict:
        """
            Returns player positions for a single frame with mapped player IDs.
            Returns:
                dict:
                    {
                        "home": [{"id": int, "x": float, "y": float}, ...],
                        "away": [{"id": int, "x": float, "y": float}, ...]
                    }
        Note: x,y normalized from -1 to 1
        """
        pass

    @abstractmethod
    def get_ball_position(self, frame: dict) -> dict:
        """
            Returns ball position for a single frame.
            Returns:
                dict: {'x': float, 'y': float, 'z': float}
            Note: x,y normalized from -1 to 1 and z normalized from 0 to 1
        """
        pass

    @abstractmethod
    def get_event_type(self, frame: dict) -> dict:
        """
            Returns:
                dict:
                    {
                        'game_event_type': int,        # GameEventType.value
                        'home_ball': bool,
                        'possession_event_type': int,  # PossessionEventType.value
                    }
        """
        pass

    @abstractmethod
    def get_time(self, frame: dict) -> dict:
        """
        Returns:
            dict:
                {'Time': float, 'Period': int}
            Time: current time from first kick off in seconds
            Period: Current half (1,2) or extra time(3,4)
        """
        pass

    @abstractmethod
    def get_sequence_id(self, frame: dict) -> int:
        """
        Returns:
            seq_id -> int
        """
        pass

    @abstractmethod
    def get_match_id(self):
        """
        Returns:
            game_id -> int
        """
        pass

    def build_sequences(self) -> dict:
        """
        Returns:
            {
                seq_id: {
                    frame_num: {
                        'player_position': ...,
                        'ball_position': ...,
                        'time': ...,
                        'event_type': ...,
                    },
                    ...
                    'total_num_frames': <int>,
                }
            }
        """
        sequences = {}
        for frame_num, frame in self.frames.items():
            seq_id = self.get_sequence_id(frame)
            sequences.setdefault(seq_id, {})[frame_num] = {
                'player_position': self.get_players_position(frame),
                'ball_position': self.get_ball_position(frame),
                'time': self.get_time(frame),
                'event_type': self.get_event_type(frame),
            }

        for seq_frames in sequences.values():
            seq_frames['total_num_frames'] = len(seq_frames)

        return sequences