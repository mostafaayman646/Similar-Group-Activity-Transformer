from abc import ABC, abstractmethod


class Data_Pipeline(ABC):
    @abstractmethod
    def get_players_position(self) -> dict:
        """
            Returns player positions indexed by frame number with mapped player IDs.
            Returns:
                dict: Mapping of frame_num to team position dicts:
                    {
                        frame_num: {
                            "home": [{"id": int, "x": float, "y": float}, ...],
                            "away": [{"id": int, "x": float, "y": float}, ...]
                        }
                    }
        Note: x,y normalized from -1 to 1
        """
        pass

    @abstractmethod
    def get_ball_position(self) -> dict:
        """frameNum -> raw ball position entry/entries for that frame."""
        pass

    @abstractmethod
    def get_event_type(self) -> dict:
        """frameNum -> {'game_event': ..., 'possession_event': ...}, raw event info for that frame."""
        pass

    @abstractmethod
    def get_time(self) -> dict:
        """frameNum -> raw timing info for that frame (videoTimeMs, period, etc.)."""
        pass

    @abstractmethod
    def get_sequence_id(self) -> dict:
        """frameNum -> sequence_id, the uninterrupted possession this frame belongs to."""
        pass

    @abstractmethod
    def get_match_id(self):
        """Unique identifier for the match this pipeline was loaded from."""
        pass