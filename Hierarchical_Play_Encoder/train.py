import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import os

# Import your custom modules
from dataset import FIFASequenceDataset
from Hierarchical_Play_Encoder.model import HierarchicalPlayEncoder

# Hyperparameters & Setup
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.1
DATA_DIR = '/kaggle/input/datasets/mostafa646/fifa-world-cup'

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Training on device: {device}")


# The Football Augmentation Engine
def augment_play(coords, p_flip_y=0.5, p_mask_player=0.15):
    """
    Creates a 'Positive Match' by mirroring and adding noise to the play.
    coords shape: [Batch, Frames, Agents, 2]
    """
    aug_coords = coords.clone()
    B, S, A, _ = aug_coords.shape

    # A. Tactical Mirroring (Y-Axis Flip)
    # 50% chance to flip the entire play vertically across the pitch
    flip_mask = torch.rand(B) < p_flip_y
    aug_coords[flip_mask, :, :, 1] = aug_coords[flip_mask, :, :, 1] * -1.0

    # B. Spatial Jitter (Camera / Tracking Noise)
    # Add tiny random noise to perfectly simulate imperfect tracking data
    noise = torch.randn_like(aug_coords) * 0.02
    aug_coords = aug_coords + noise

    # C. Agent Dropout (Red Card / Missing Data effect)
    # Randomly drop a player out of the sequence, but keep the ball (index 22) safe
    player_mask = torch.rand(B, 1, A, 1) > p_mask_player
    player_mask[:, :, 22, :] = True  # Index 22 is the Ball

    # Apply mask and clamp coordinates to stay within the normalized pitch [-1.0, 1.0]
    aug_coords = aug_coords * player_mask.to(aug_coords.device)
    return torch.clamp(aug_coords, min=-1.0, max=1.0)



def main():
    Num_workers = 3

    print("Loading Dataset")
    dataset = FIFASequenceDataset(data_dir=DATA_DIR, target_frames=100)

    # drop_last=True ensures every batch is exactly 8 plays (required for pure contrastive math)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=Num_workers)

    print("Initializing Model")
    model = HierarchicalPlayEncoder(d_model=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("\nStarting Training\n")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # Move data to GPU
            coords = batch['coordinates'].to(device)
            roles = batch['roles'].to(device)

            optimizer.zero_grad()

            # Step A: Create the "Positive Match" view using Augmentation
            coords_view_1 = coords
            coords_view_2 = augment_play(coords)

            # Step B: Push both views through the model
            # Output shape: [Batch_Size, 128]
            embeds_1 = model(coords_view_1, roles)
            embeds_2 = model(coords_view_2, roles)

            # Step C: L2 Normalize the vectors (Forces them onto a unit sphere)
            embeds_1 = F.normalize(embeds_1, p=2, dim=1)
            embeds_2 = F.normalize(embeds_2, p=2, dim=1)

            # Step D: InfoNCE Contrastive Loss
            # Matrix multiplication calculates similarity of every play against every augmented play
            logits = torch.matmul(embeds_1, embeds_2.T) / TEMPERATURE

            # The correct matches are the diagonal (e.g., Play 0 matches Aug_Play 0)
            labels = torch.arange(BATCH_SIZE).to(device)

            loss = F.cross_entropy(logits, labels)

            # Step E: Backpropagate and Optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Print progress
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Contrastive Loss: {avg_loss:.4f}")

    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining Complete in {elapsed:.2f} minutes")

    # Save the Baseline Weights
    torch.save(model.state_dict(), "temporal_baseline.pth")
    print("Saved model weights to 'temporal_baseline.pth'")


if __name__ == "__main__":
    main()