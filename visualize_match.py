"""
visualize_match.py

Simple 2D visualizer for football tracking sequences produced by the
Football Data Pipeline (see README.md / Data_interface.py in this project).

It reads the per-match files written by main.py (one gzip-compressed
torch.save blob per match, named match_<id>.pt.gz, inside a "matches"
folder) and animates the players + ball for the first `k` sequences of a
chosen match on a 2D pitch.

Usage
-----
    python visualize_match.py --pt_path "FIFA World Cup 2022/matches" \
                               --match_id 12345 \
                               --k 3

Or just edit the DEFAULT_* constants below and run:
    python visualize_match.py

By default the animation plays live in a matplotlib window. Pass
--save out.mp4 (needs ffmpeg) or --save out.gif (needs pillow) to write it
to disk instead.
"""

import argparse
import gzip
import io
import shutil
import struct
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg') # Tell Matplotlib to run in headless mode
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# Defaults -- edit these if you'd rather not use the CLI
# ---------------------------------------------------------------------------
DEFAULT_PT_PATH = "FIFA World Cup 2022/matches"
DEFAULT_MATCH_ID = 10510      # None -> uses the first match found in the file
DEFAULT_K = 3                # number of sequences to play
DEFAULT_INTERVAL_MS = 60     # delay between frames (ms) -> lower = faster
DEFAULT_SAVE_PATH = None     # e.g. "match.mp4" / "match.gif", or 'auto' for a generated filename

_HEADER = struct.Struct('<Q')  # matches main.py's on-disk record format


# ---------------------------------------------------------------------------
# Loading -- main.py now writes one match_*.pt.gz file per match inside a
# "matches" directory, so we only ever need to open the one file we want
# (or, if no --match_id is given, the first file in the directory). This
# avoids ever materializing the whole tournament in RAM, which is what
# caused earlier versions to run out of memory.
#
# A legacy fallback is kept for the old single-file format (an
# [8-byte length][gzip(torch.save blob)] stream) in case you still have an
# all_matches.pt lying around from before.
# ---------------------------------------------------------------------------
def _ids_match(seen_id, requested_id) -> bool:
    if seen_id == requested_id:
        return True
    # ids may be stored as int on disk but passed as a string on the CLI
    return str(seen_id) == str(requested_id)


def _load_match_file(file_path: Path) -> dict:
    """Loads a single match_*.pt.gz file -> {match_id: match_data}."""
    raw = gzip.decompress(file_path.read_bytes())
    return torch.load(io.BytesIO(raw), weights_only=False)


def find_match(path, match_id=None):
    """
    Locates a single match and returns (matched_id, match_data).

    `path` may be:
      - a directory of match_*.pt.gz files (the current main.py output), or
      - a single legacy .pt file containing the old
        [8-byte length][gzip blob] stream format.

    If match_id is given, returns the first match found with that id, then
    stops reading further files/records. If match_id is None, returns the
    first match found. If not found, raises KeyError listing the match_ids
    that were seen (ids only -- their frame data is never kept, so this
    stays cheap even for a full scan).
    """
    path = Path(path)

    if path.is_dir():
        # Fast path: if we know the id, try the exact filename directly.
        if match_id is not None:
            direct = path / f"match_{match_id}.pt.gz"
            if direct.exists():
                chunk = _load_match_file(direct)
                mid, mdata = next(iter(chunk.items()))
                return mid, mdata

        seen_ids = []
        for file_path in sorted(path.glob('match_*.pt.gz')):
            chunk = _load_match_file(file_path)
            for mid, mdata in chunk.items():
                if match_id is None:
                    return mid, mdata
                if _ids_match(mid, match_id):
                    return mid, mdata
                seen_ids.append(mid)

        raise KeyError(
            f"match_id {match_id!r} not found in {path}. "
            f"Match ids seen: {seen_ids[:10]}{'...' if len(seen_ids) > 10 else ''}"
        )

    # Legacy single-file stream format.
    seen_ids = []
    with open(path, 'rb') as f:
        while True:
            header = f.read(_HEADER.size)
            if not header:
                break
            (n,) = _HEADER.unpack(header)
            raw = gzip.decompress(f.read(n))
            chunk = torch.load(io.BytesIO(raw), weights_only=False)
            del raw

            for mid, mdata in chunk.items():
                if match_id is None:
                    return mid, mdata
                if _ids_match(mid, match_id):
                    return mid, mdata
                seen_ids.append(mid)
            # not the match we wanted -- drop this whole record before
            # reading the next one
            del chunk

    raise KeyError(
        f"match_id {match_id!r} not found in {path}. "
        f"Match ids seen: {seen_ids[:10]}{'...' if len(seen_ids) > 10 else ''}"
    )


def collect_frames(match_data: dict, k: int):
    """
    Flattens the first `k` sequences of a single match into an ordered list
    of (seq_id, frame_num, frame_dict). Each sequence dict mixes frame_num
    keys with a 'total_num_frames' key at the same level, so that key is
    skipped here.
    """
    seq_ids = list(match_data.keys())[:k]
    frames = []
    for seq_id in seq_ids:
        seq = match_data[seq_id]
        frame_nums = sorted(fn for fn in seq.keys() if fn != 'total_num_frames')
        for fn in frame_nums:
            frames.append((seq_id, fn, seq[fn]))
    return frames, seq_ids


# ---------------------------------------------------------------------------
# Pitch drawing
# ---------------------------------------------------------------------------
def draw_pitch(ax):
    """Draws a simple pitch using the -1..1 normalized coordinate system
    that build_sequences() outputs positions in."""
    ax.set_facecolor('#2e7d32')  # grass green

    # Outer boundary
    ax.add_patch(patches.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='white', linewidth=1.5))
    # Halfway line
    ax.plot([0, 0], [-1, 1], color='white', linewidth=1)
    # Center circle
    ax.add_patch(patches.Circle((0, 0), 0.15, fill=False, edgecolor='white', linewidth=1))
    ax.plot(0, 0, marker='o', color='white', markersize=2)
    # Penalty boxes (rough proportions)
    box_w, box_h = 0.16, 0.6
    ax.add_patch(patches.Rectangle((-1, -box_h / 2), box_w, box_h, fill=False, edgecolor='white', linewidth=1))
    ax.add_patch(patches.Rectangle((1 - box_w, -box_h / 2), box_w, box_h, fill=False, edgecolor='white', linewidth=1))
    # Goals
    goal_h = 0.12
    ax.add_patch(patches.Rectangle((-1.02, -goal_h / 2), 0.02, goal_h, fill=True, facecolor='white'))
    ax.add_patch(patches.Rectangle((1.0, -goal_h / 2), 0.02, goal_h, fill=True, facecolor='white'))

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect(68 / 105)  # real pitch proportions (~68m wide x 105m long)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
def has_ffmpeg() -> bool:
    return shutil.which('ffmpeg') is not None


def resolve_save_path(save_arg, match_id, k) -> str:
    """
    --save with no value -> auto filename in the current directory.
    --save some/path      -> used as given, but the extension is corrected
                              to .gif if ffmpeg isn't available.
    """
    if save_arg in (None, 'auto'):
        path = f"match_{match_id}_first{k}seqs.mp4"
    else:
        path = save_arg

    if not path.endswith('.gif') and not has_ffmpeg():
        print("ffmpeg not found on this system -- saving as .gif instead (needs Pillow).")
        path = str(Path(path).with_suffix('.gif'))

    return path


def animate_match(match_data: dict, match_id, k: int, interval: int, save_path: str = None, show: bool = True):
    frames, seq_ids = collect_frames(match_data, k)
    if not frames:
        print(f"No frames found for match {match_id} (k={k}).")
        return

    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor('#1b1b1b')
    draw_pitch(ax)

    home_scatter = ax.scatter([], [], s=90, c='#e53935', edgecolors='white', linewidths=0.8, zorder=3, label='Home')
    away_scatter = ax.scatter([], [], s=90, c='#1e88e5', edgecolors='white', linewidths=0.8, zorder=3, label='Away')
    ball_scatter = ax.scatter([], [], s=35, c='#fdd835', edgecolors='black', linewidths=0.6, zorder=4, label='Ball')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, labelcolor='white')

    title = ax.set_title('', color='white', fontsize=10)

    def update(i):
        seq_id, frame_num, frame = frames[i]

        home = frame['player_position'].get('home', [])
        away = frame['player_position'].get('away', [])
        ball = frame['ball_position']
        t = frame.get('time', {})

        home_scatter.set_offsets([[p['x'], p['y']] for p in home])
        away_scatter.set_offsets([[p['x'], p['y']] for p in away])
        ball_scatter.set_offsets([[ball['x'], ball['y']]])

        title.set_text(
            f"Match {match_id} | Seq {seq_id} | Frame {frame_num} | "
            f"Period {t.get('Period', '?')} | Time {t.get('Time', 0):.1f}s"
        )
        return home_scatter, away_scatter, ball_scatter, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=interval, blit=False, repeat=True
    )

    if save_path:
        fps = max(1, round(1000 / interval))
        total = len(frames)

        def _progress(current_frame, total_frames):
            print(f"\rRendering {save_path}: {current_frame + 1}/{total_frames}", end='', flush=True)

        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps, progress_callback=_progress)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps, progress_callback=_progress)
        print(f"\nSaved animation to {save_path} ({total} frames) -- reopen this file any time, no need to rerun the script.")

    if show and not save_path:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize player/ball movement for a match.")
    parser.add_argument('--pt_path', type=str, default=DEFAULT_PT_PATH,
                         help='Directory of per-match match_*.pt.gz files written by main.py '
                              '(or a legacy single all_matches.pt file).')
    parser.add_argument('--match_id', type=str, default=DEFAULT_MATCH_ID)
    parser.add_argument('--k', type=int, default=DEFAULT_K, help='Number of sequences to visualize')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL_MS, help='ms between frames')
    parser.add_argument(
        '--save', nargs='?', const='auto', default=DEFAULT_SAVE_PATH,
        help='Save the animation to a file instead of just showing it live. '
             'Use "--save" alone for an auto-generated filename, or "--save path.mp4" / "--save path.gif" '
             'for a specific one. Falls back to .gif automatically if ffmpeg is not installed.'
    )
    parser.add_argument('--show', action='store_true', help='Also open the live window even when --save is used.')
    args = parser.parse_args()

    path = Path(args.pt_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Pass --pt_path to point at your all_matches.pt file.")

    if args.match_id is None:
        print(f"No --match_id given, scanning {path} for the first match...")
    else:
        print(f"Scanning {path} for match_id={args.match_id} ...")

    match_id, match_data = find_match(path, args.match_id)
    print(f"Found match {match_id} ({len(match_data)} sequences total).")

    save_path = resolve_save_path(args.save, match_id, args.k) if args.save else None
    animate_match(match_data, match_id, args.k, args.interval, save_path, show=args.show)


if __name__ == '__main__':
    main()