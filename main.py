import torch
from pathlib import Path
from Data import build_sequences
from Data import Dataset

all_matches = {}
data_dir = 'FIFA World Cup 2022'
counter = 0
for path in Path(f'{data_dir}/Tracking Data').glob('*.jsonl.bz2'):
    if counter == 5:
        break
    json_id = path.name.split('.')[0]  # Extracts the ID from filename (e.g., '10510.jsonl.bz2' -> '10510')
    match_data = build_sequences(
        dataset=Dataset.FIFA_WC_2022,
        data_dir_path=data_dir,
        json_id=json_id
    )
    all_matches.update(match_data)
    
    counter+=1
    print(f"Finished:{counter}")

torch.save(all_matches, f'{data_dir}/all_matches.pt')