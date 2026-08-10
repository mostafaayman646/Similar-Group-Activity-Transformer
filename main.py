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
MAX_MATCHES = 5      # set to None to process every match found
NUM_WORKERS = None   # None -> auto (bounded by CPU count and number of matches)
COMPRESS_LEVEL = 1   # 1 = fastest / bigger file, 9 = slowest / smallest file

_HEADER = struct.Struct('<Q')  # 8-byte length prefix per stored match


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

    buf = io.BytesIO()
    torch.save(match_data, buf, pickle_protocol=5)
    compressed = gzip.compress(buf.getvalue(), compresslevel=COMPRESS_LEVEL)
    return json_id, compressed


def main():
    tracking_dir = Path(DATA_DIR) / 'Tracking Data'
    json_ids = [p.name.split('.')[0] for p in tracking_dir.glob('*.jsonl.bz2')]
    if MAX_MATCHES is not None:
        json_ids = json_ids[:MAX_MATCHES]

    if not json_ids:
        print("No matches found.")
        return

    num_workers = NUM_WORKERS or min(len(json_ids), os.cpu_count() or 1)
    out_path = Path(DATA_DIR) / 'all_matches.pt'

    t0 = time.time()
    saved = 0
    with open(out_path, 'wb') as out_f, ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_build_and_pack, DATA_DIR, jid): jid for jid in json_ids}
        for i, future in enumerate(as_completed(futures), 1):
            json_id, compressed = future.result()
            if compressed is not None:
                out_f.write(_HEADER.pack(len(compressed)))
                out_f.write(compressed)
                saved += 1
            print(f"Finished {i}/{len(json_ids)} ({json_id})")

    dt = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {saved}/{len(json_ids)} matches -> {out_path} ({size_mb:.1f} MB) in {dt:.1f}s")


def load_all_matches(path=None) -> dict:
    """Reads back the file written by main(): a sequence of
    [8-byte length][gzip-compressed torch.save blob] records, one per match."""
    path = Path(path) if path else Path(DATA_DIR) / 'all_matches.pt'
    all_matches = {}
    with open(path, 'rb') as f:
        while True:
            header = f.read(_HEADER.size)
            if not header:
                break
            (n,) = _HEADER.unpack(header)
            raw = gzip.decompress(f.read(n))
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