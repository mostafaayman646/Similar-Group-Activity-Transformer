import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

# Import your custom modules
from data import FIFASequenceDataset
from Hierarchical_Play_Encoder.model import HierarchicalPlayEncoder
from utils import setup_logger, save_checkpoint, load_config


def augment_play(coords, p_flip_y=0.5, p_mask_player=0.15):
    """ Creates a 'Positive Match' by mirroring and adding noise. """
    aug_coords = coords.clone()
    B, S, A, _ = aug_coords.shape

    # Tactical Mirroring
    flip_mask = torch.rand(B) < p_flip_y
    aug_coords[flip_mask, :, :, 1] = aug_coords[flip_mask, :, :, 1] * -1.0

    # Spatial Jitter
    noise = torch.randn_like(aug_coords) * 0.02
    aug_coords = aug_coords + noise

    # Agent Dropout (protect index 22 = the ball)
    player_mask = torch.rand(B, 1, A, 1) > p_mask_player
    player_mask[:, :, 22, :] = True

    aug_coords = aug_coords * player_mask.to(aug_coords.device)
    return torch.clamp(aug_coords, min=-1.0, max=1.0)

def main():
    # Load configuration
    config = load_config("Configs/hier_model_config.yaml")

    # Setup components using config values
    logger = setup_logger(config['logging']['log_file'])
    logger.info("Initializing Training Pipeline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Dataset & DataLoader
    logger.info("Loading Dataset")
    dataset = FIFASequenceDataset(
        data_dir=config['data']['data_dir'],
        target_frames=config['data']['target_frames']
    )
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, drop_last=True)

    # Model & Optimizer
    logger.info("Initializing Hierarchical Model")
    model = HierarchicalPlayEncoder(
        d_model=config['model']['embed_dim'],
        n_heads=config['model']['n_heads'],
        frame_layers=config['model']['frame_layers'],
        play_layers=config['model']['play_layers']
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )

    logger.info("\nStarting Training\n")
    best_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            coords = batch['coordinates'].to(device)
            roles = batch['roles'].to(device)

            optimizer.zero_grad()

            # Augmentation
            coords_view_1 = coords
            coords_view_2 = augment_play(coords)

            # Model Forward Pass
            try:
                embeds_1 = model(coords_view_1, roles)
                embeds_2 = model(coords_view_2, roles)
            except Exception as e:
                logger.error(f"Forward pass failed at batch {batch_idx}: {str(e)}")
                break

            # L2 Normalization
            embeds_1 = F.normalize(embeds_1, p=2, dim=1)
            embeds_2 = F.normalize(embeds_2, p=2, dim=1)

            # InfoNCE Loss
            logits = torch.matmul(embeds_1, embeds_2.T) / config['training']['temperature']
            labels = torch.arange(config['training']['batch_size']).to(device)
            loss = F.cross_entropy(logits, labels)

            # Backward and Optimize
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['clip_max_norm'])

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Check if this is our best epoch so far
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # Log epoch results
        logger.info(
            f"Epoch [{epoch + 1}/{config['training']['epochs']}] | Contrastive Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'config': config
        }
        save_checkpoint(checkpoint, is_best, config['logging']['checkpoint_dir'])



if __name__ == "__main__":
    main()