from abc import ABC, abstractmethod


class Data_Pipeline(ABC):
    @abstractmethod
    def get_players_position(self) -> dict:
        """frameNum -> {"home": [...], "away": [...]} raw player position entries."""
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