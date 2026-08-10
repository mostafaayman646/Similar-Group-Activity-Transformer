# Football Data Pipeline

A common interface for turning raw, provider-specific tracking/event data (e.g. PFF FC,
Sportlogiq, StatsPerform, ...) into a single, dataset-agnostic sequence format that the rest of
the codebase can consume without caring where the data came from.

## How it works

```
Data/
├── Data_interface.py   # Abstract base class every provider must implement
├── Data_factory.py     # Dispatches to the right provider and builds sequences
├── Dataset_enum.py     # Registry of supported datasets + shared event-type enums
├── README.md
└── Providers/
    ├── __init__.py
    └── FIFAWC22.py      # Example provider (PFF FC / FIFA World Cup 2022 data)
```

**`Data_interface.py`** defines `Data_Pipeline`, an abstract class. Every dataset provider
(e.g. `FIFAWC22`) must subclass it and implement all abstract methods. This is the contract
that guarantees any provider can be plugged into the rest of the pipeline interchangeably.

**`Data_factory.py`** exposes `build_sequences(dataset, data_dir_path, json_id)`, the single
entry point used by the rest of the codebase. Given a `Dataset` enum value, it instantiates the
matching provider, calls its interface methods, and assembles everything into one nested
sequence dictionary — the caller never touches provider-specific logic directly.

**`Dataset_enum.py`** is the registry: it lists every dataset the factory knows how to build,
plus the shared `GameEventType` and `PossessionEventType` enums that every provider must map
its raw event strings onto (see below).

## Output format

`build_sequences(...)` always returns:

```python
{
    match_id: {
        seq_id: {
            frame_num: {
                'player_position': {'home': [{'id': int, 'x': float, 'y': float}, ...],
                                     'away': [{'id': int, 'x': float, 'y': float}, ...]},
                'ball_position': {'x': float, 'y': float, 'z': float},
                'time': {'Time': float, 'Period': int},
                'event_type': {'game_event_type': str, 'home_ball': bool,
                                'possession_event_type': str},
            },
            ...
            'total_num_frames': int,
        },
        ...
    }
}
```

Every provider must produce data in exactly this shape (see `Data_interface.py` docstrings for
per-method details) regardless of what the raw source data looks like. That's what makes
providers interchangeable.

## Adding a new dataset

Follow these steps to plug in a new data source without touching any existing provider or the
factory's dispatch contract:

### 1. Register the dataset

Add a new member to `Dataset` in `Dataset_enum.py`:

```python
class Dataset(Enum):
    FIFA_WC_2022 = 'FIFA_WC_2022'
    MY_NEW_DATASET = 'MY_NEW_DATASET'
```

### 2. Create the provider

Add a new file under `Data/Providers/`, e.g. `Data/Providers/MyNewDataset.py`, and implement a
class that subclasses `Data_Pipeline` and implements **every** abstract method:

- `get_players_position() -> dict`
- `get_ball_position() -> dict`
- `get_event_type() -> dict`
- `get_time() -> dict`
- `get_sequence_id() -> dict`
- `get_match_id()`

Read the docstrings in `Data_interface.py` carefully — they specify the exact return shape and
value ranges (e.g. positions normalized to `-1..1`, `z` normalized to `0..1`) expected from each
method. Frame filtering, coordinate normalization, sequence-id assignment, and any
forward/backward filling of missing values are the provider's responsibility, not the
factory's.

**Event-type mapping is mandatory.** `get_event_type()` must return values from the shared
`GameEventType` / `PossessionEventType` enums defined in `Dataset_enum.py`, not raw strings from
your source data. Map your provider's raw event codes onto these enums so downstream code never
needs to know which dataset a sequence came from. If your source has an event with no direct
equivalent, do not invent a new global category — either map it to the closest existing enum
member or raise an issue to discuss extending the shared enum (extending it affects every
provider, so it should be a deliberate, reviewed change).

### 3. Wire it into the factory

In `Data_factory.py`, import your new provider and add a branch to the dispatcher:

```python
from Data.Providers import FIFAWC22, MyNewDataset

if dataset == Dataset.FIFA_WC_2022:
    pipeline = FIFAWC22(data_dir_path, json_id)
elif dataset == Dataset.MY_NEW_DATASET:
    pipeline = MyNewDataset(data_dir_path, json_id)
else:
    raise NotImplementedError(f"No adapter registered for dataset: {dataset}")
```

Nothing else in `build_sequences` needs to change — it drives every provider through the same
`Data_Pipeline` interface.

### 4. Sanity-check your provider

Before considering it done, confirm for a sample match:

- Every method returns data keyed by the same `frame_num`s used by `get_sequence_id()`.
- `player_position` / `ball_position` coordinates are normalized as documented.
- `event_type` values only use members of `GameEventType` / `PossessionEventType`.
- `get_match_id()` returns a stable, unique identifier for the match.
- Frames with missing/unreliable tracking (e.g. low confidence, no ball, missing video) are
  filtered out consistently with how `FIFAWC22` does it, or documented if your source doesn't
  need this.

## Shared event-type enums

Because different providers label events differently, every provider must translate its raw
event codes into these two shared enums so that consumers of `build_sequences` can rely on a
single vocabulary regardless of the underlying dataset.

### `GameEventType`

| Member | Meaning |
|---|---|
| `FIRSTKICKOFF` | First half kick-off |
| `SECONDKICKOFF` | Second half kick-off |
| `END` | End of half |
| `G` | Ball hits post, bar, or corner flag and stays in play |
| `OFF` | Player off |
| `ON` | Player on |
| `OTB` | On-the-ball event |
| `OUT` | Ball out-of-play |
| `SUB` | Substitution |
| `NOTB` | Not on the ball (ball moving between feet of the same player in control) |

### `PossessionEventType`

| Member | Meaning |
|---|---|
| `BC` | Ball carry |
| `CH` | Challenge |
| `CL` | Clearance |
| `CR` | Cross |
| `PA` | Pass |
| `RE` | Rebound |
| `SH` | Shot |
| `DR` | Dribbling |

See `Dataset_enum.py` for the exact enum definitions and integer values.
