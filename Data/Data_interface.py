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
        """
            Returns ball position indexed by frame number.
            Returns:
                dict:
                    {
                        frame_num: {'x': float, 'y': float, 'z': float}
                    }
        Note: x,y normalized from -1 to 1 and z normalized from 0 to 1
        """
        pass

    @abstractmethod
    def get_event_type(self) -> dict:
        """
            Returns events indexed by frame number.
            Returns:
                dict:
                    {
                        frame_num: {'game_event_type': str, 'home_ball': bool, 'possession_event_type': str}
                    }
        """
        pass

    @abstractmethod
    def get_time(self) -> dict:
        """
        Returns:
            dict:
                {
                    frame_num: {'Time': float, 'Period': int}
                }.
        Time:current time from first kick off in seconds
        Period: Current half (1,2) or extra time(3,4)
        """
        pass

    @abstractmethod
    def get_sequence_id(self) -> dict:
        """frameNum -> sequence_id, the uninterrupted possession this frame belongs to."""
        pass

    @abstractmethod
    def get_match_id(self):
        """Unique identifier for the match this pipeline was loaded from."""
        pass