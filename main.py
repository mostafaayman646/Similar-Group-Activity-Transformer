import torch
from pathlib import Path
import itertools
from Data import build_sequences
from Data import Dataset

fifa_wc_22_dict = build_sequences(
    dataset=Dataset.FIFA_WC_2022,
    data_dir_path='FIFA World Cup 2022',
    json_id='10510'
)

first_match_id = next(iter(fifa_wc_22_dict))
first_match = fifa_wc_22_dict[first_match_id]

first_seq_id = next(iter(first_match))
first_sequence = first_match[first_seq_id]

first_frame = dict(itertools.islice(first_sequence.items(), 1))

print(f"match_id: {first_match_id}")
print(f"sequence_id: {first_seq_id}")
print(first_frame)


total_frames = 0
print("\ntotal_num_frames per sequence:")
for seq_id, seq_frames in first_match.items():
    total_frames += seq_frames['total_num_frames']
    print(f"  sequence {seq_id}: {seq_frames['total_num_frames']}")


print(f"\nTotal Frames:{total_frames}")
# all_matches = {}
# for path in Path('FIFA World Cup 2022/Tracking Data').glob('*.jsonl.bz2'):
#     all_matches.update(build_sequences(Dataset.FIFA_WC_2022, str(path)))

# torch.save(all_matches, 'all_matches.pt')


# Frames:170717
# Frames:100124