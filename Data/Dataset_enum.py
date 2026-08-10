from enum import Enum


class Dataset(Enum):
    FIFA_WC_2022 = 'FIFA_WC_2022'


class GameEventType(Enum):
    """
    Shared game event vocabulary. Every dataset provider must map its raw
    game-event codes onto these members before returning data from
    get_event_type().
    """
    FIRSTKICKOFF = 0    # First half kick-off
    SECONDKICKOFF = 1   # Second half kick-off
    END = 2              # End of half
    G = 3                # Ball hits post, bar, or corner flag and stays in play
    OFF = 4              # Player off
    ON = 5               # Player on
    OTB = 6               # On-the-ball event
    OUT = 7               # Ball out-of-play
    SUB = 8               # Substitution
    NOTB = 9              # Not on the ball (ball moving between feet of same player)


class PossessionEventType(Enum):
    """
    Shared possession event vocabulary. Every dataset provider must map its
    raw possession-event codes onto these members before returning data from
    get_event_type().
    """
    BC = 0   # Ball carry
    CH = 1   # Challenge
    CL = 2   # Clearance
    CR = 3   # Cross
    PA = 4   # Pass
    RE = 5   # Rebound
    SH = 6   # Shot
    DR = 7   # Dribbling
