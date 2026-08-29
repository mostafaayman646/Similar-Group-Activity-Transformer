import gzip
import io
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import itertools

import torch

from Data import build_sequences
from Data import Dataset

DATA_DIR = 'FIFA World Cup 2022'
MATCHES_SUBDIR = 'matches'   # per-match files are written here, one file per match
MAX_MATCHES = None      # set to None to process every match found
NUM_WORKERS = 3   # None -> auto (bounded by CPU count and number of matches)
COMPRESS_LEVEL = 1   # 1 = fastest / bigger file, 9 = slowest / smallest file


def _build_and_pack(data_dir: str, json_id: str):
    try:
        match_data = build_sequences(
            dataset=Dataset.FIFA_WC_2022,
            data_dir_path=data_dir,
            json_id=json_id,
        )
    except Exception as exc:
        print(f"Skipped {json_id}: {exc!r}")
        return json_id, None

    # build_sequences() already returns {json_id: {seq_id: {...}, ...}}, so
    # save it as-is -- do NOT wrap it in another {json_id: match_data}, or
    # every file ends up double-nested (match_data[json_id] instead of the
    # real per-sequence dict).
    buf = io.BytesIO()
    torch.save(match_data, buf, pickle_protocol=5)
    compressed = gzip.compress(buf.getvalue(), compresslevel=COMPRESS_LEVEL)
    return json_id, compressed


def _match_out_path(matches_dir: Path, json_id) -> Path:
    """Filesystem-safe path for a single match's file."""
    safe_id = str(json_id).replace('/', '_').replace('\\', '_')
    return matches_dir / f"match_{safe_id}.pt.gz"


def main():
    tracking_dir = Path(DATA_DIR) / 'Tracking Data'
    json_ids = [p.name.split('.')[0] for p in tracking_dir.glob('*.jsonl.bz2')]
    if MAX_MATCHES is not None:
        json_ids = json_ids[:MAX_MATCHES]

    if not json_ids:
        print("No matches found.")
        return

    num_workers = NUM_WORKERS or min(len(json_ids), os.cpu_count() or 1)
    matches_dir = Path(DATA_DIR) / MATCHES_SUBDIR
    matches_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    saved = 0
    total_bytes = 0
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_build_and_pack, DATA_DIR, jid): jid for jid in json_ids}
        for i, future in enumerate(as_completed(futures), 1):
            json_id, compressed = future.result()
            if compressed is not None:
                out_path = _match_out_path(matches_dir, json_id)
                out_path.write_bytes(compressed)
                saved += 1
                total_bytes += len(compressed)
            print(f"Finished {i}/{len(json_ids)} ({json_id})")

    dt = time.time() - t0
    size_mb = total_bytes / (1024 * 1024)
    print(f"Saved {saved}/{len(json_ids)} matches -> {matches_dir}/ ({size_mb:.1f} MB total) in {dt:.1f}s")


def load_match(json_id, matches_dir=None) -> dict:
    """Loads a single match by id from its own file. Returns {json_id: match_data}."""
    matches_dir = Path(matches_dir) if matches_dir else Path(DATA_DIR) / MATCHES_SUBDIR
    path = _match_out_path(matches_dir, json_id)
    if not path.exists():
        raise FileNotFoundError(f"No saved match file for {json_id!r} at {path}")
    raw = gzip.decompress(path.read_bytes())
    return torch.load(io.BytesIO(raw), weights_only=False)


def load_all_matches(matches_dir=None) -> dict:
    """Reads back every per-match file written by main() (one match_*.pt.gz
    file per match) and merges them into a single {match_id: match_data} dict."""
    matches_dir = Path(matches_dir) if matches_dir else Path(DATA_DIR) / MATCHES_SUBDIR
    all_matches = {}
    for path in sorted(Path(matches_dir).glob('match_*.pt.gz')):
        raw = gzip.decompress(path.read_bytes())
        chunk = torch.load(io.BytesIO(raw), weights_only=False)
        all_matches.update(chunk)
    return all_matches


if __name__ == '__main__':
    main()
    # matches = load_all_matches()

    # first_match_id = next(iter(matches))
    # first_match = matches[first_match_id]

    # first_seq_id = next(iter(first_match))
    # first_sequence = first_match[first_seq_id]

    # first_5_frame = dict(itertools.islice(first_sequence.items(), 5))

    # print(f"match_id: {first_match_id}")
    # print(f"sequence_id: {first_seq_id}")
    # print(first_5_frame)


    # total_frames = 0
    # print("\ntotal_num_frames per sequence:")
    # for seq_id, seq_frames in first_match.items():
    #     total_frames += seq_frames['total_num_frames']
    #     print(f"  sequence {seq_id}: {seq_frames['total_num_frames']}")


    # print(f"\nTotal Frames:{total_frames}")