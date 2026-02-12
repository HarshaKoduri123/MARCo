import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from config import config
from model.MARCo_v2 import MARCo, get_mask
from data import load_dfc_data, create_data_loaders, create_labeled_data_loaders
from utils import set_seed, save_checkpoint, load_checkpoint, save_training_history
from evaluation import (
    evaluate_classification, train_linear_segmentation_probe, evaluate_segmentation_probe, extract_representations, compute_umap_embeddings, pixel_to_image_labels
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config, 
                 train_images, train_labels, val_images, val_labels):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels
        

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            epochs=config.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        self.scaler = GradScaler()

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(config.LOG_DIR, f'train_{current_time}'))

        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.evaluation_results = {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(config.RESULTS_DIR, self.run_id)

        self.plots_dir = os.path.join(self.run_dir, "plots")
        self.tables_dir = os.path.join(self.run_dir, "tables")
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        self.embeddings_dir = os.path.join(self.run_dir, "embeddings")

        for d in [self.run_dir, self.plots_dir, self.tables_dir, self.metrics_dir, self.embeddings_dir]:
            os.makedirs(d, exist_ok=True)

    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_contrast = 0
        total_mae = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device).float()

            batch_size = images.shape[0]

            shared_mask_info = get_mask(
                bsz=batch_size,
                seq_len=self.config.NUM_PATCHES,
                device=self.device,
                mask_ratio=self.config.MASK_RATIO
            )

            radar_mask_info = shared_mask_info
            optical_mask_info = shared_mask_info

            with autocast(device_type=self.device.type):
                contrast_loss, mae_loss, ssl_loss  = self.model(
                    imgs=images,
                    radar_mask_info=radar_mask_info,
                    optical_mask_info=optical_mask_info,
                    rank=self.config.RANK,
                    world_size=self.config.WORLD_SIZE
                )
                
                total_loss_combined = (
                    self.config.CONTRAST_WEIGHT * contrast_loss +
                    self.config.MAE_WEIGHT * mae_loss +
                    self.config.SSL_WEIGHT * ssl_loss
                )



            self.optimizer.zero_grad()
            self.scaler.scale(total_loss_combined).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
 
            total_loss += total_loss_combined.item()
            total_contrast += contrast_loss.item()
            total_mae += mae_loss.item()

            pbar.set_postfix({
                'loss': total_loss_combined.item(),
                'contrast': contrast_loss.item(),
                'mae': mae_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })

            if batch_idx % self.config.LOG_FREQUENCY == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss', total_loss_combined.item(), step)
                self.writer.add_scalar('train/contrast_loss', contrast_loss.item(), step)
                self.writer.add_scalar('train/mae_loss', mae_loss.item(), step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], step)

        avg_loss = total_loss / len(self.train_loader)
        avg_contrast = total_contrast / len(self.train_loader)
        avg_mae = total_mae / len(self.train_loader)
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, avg_contrast, avg_mae
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_contrast = 0
        total_mae = 0
        
        pbar = tqdm(self.val_loader, desc=f'Validation {epoch}')
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device).float()

            batch_size = images.shape[0]
            
            # Generate masks
            shared_mask_info = get_mask(
                bsz=batch_size,
                seq_len=self.config.NUM_PATCHES,
                device=self.device,
                mask_ratio=self.config.MASK_RATIO
            )

            radar_mask_info = shared_mask_info
            optical_mask_info = shared_mask_info

                        
            # Forward pass
            with autocast(device_type=self.device.type):
                contrast_loss, mae_loss, ssl_loss  = self.model(
                    imgs=images,
                    radar_mask_info=radar_mask_info,
                    optical_mask_info=optical_mask_info,
                    rank=self.config.RANK,
                    world_size=self.config.WORLD_SIZE
                )
                
                total_loss_combined = (
                    self.config.CONTRAST_WEIGHT * contrast_loss +
                    self.config.MAE_WEIGHT * mae_loss +
                    self.config.SSL_WEIGHT * ssl_loss
                )

            
            # Update metrics
            total_loss += total_loss_combined.item()
            total_contrast += contrast_loss.item()
            total_mae += mae_loss.item()
            
            pbar.set_postfix({
                'val_loss': total_loss_combined.item(),
                'val_contrast': contrast_loss.item(),
                'val_mae': mae_loss.item()
            })
        
        # Calculate averages
        avg_loss = total_loss / len(self.val_loader)
        avg_contrast = total_contrast / len(self.val_loader)
        avg_mae = total_mae / len(self.val_loader)
        
        self.val_losses.append(avg_loss)
        
        # Log to TensorBoard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/contrast_loss', avg_contrast, epoch)
        self.writer.add_scalar('val/mae_loss', avg_mae, epoch)
        
        return avg_loss, avg_contrast, avg_mae
    
    def run_evaluation(self, epoch):
        print(f"\nRunning Evaluation at Epoch {epoch}")

        num_samples = min(2000, len(self.val_images))

        train_loader, val_loader = create_labeled_data_loaders(
            self.train_images[:num_samples],
            self.train_labels[:num_samples],
            self.val_images[:num_samples],
            self.val_labels[:num_samples],
            batch_size=64
        )

 

        reps_val, y_val = extract_representations(self.model, val_loader)

        torch.save(
            {"reps": reps_val, "labels": y_val},
            os.path.join(self.embeddings_dir, f"reps_epoch_{epoch}.pt")
        )

        cls_results = {
            "linear": evaluate_classification(
                reps_val, y_val,
                task_type="single",
                method="linear",
                num_classes=8
            ),
            "mlp": evaluate_classification(
                reps_val, y_val,
                task_type="single",
                method="mlp",
                num_classes=8
            ),
            "knn": evaluate_classification(
                reps_val, y_val,
                task_type="single",
                method="knn",
                num_classes=8
            ),
            "kmeans": evaluate_classification(
                reps_val, y_val,
                task_type="single",
                method="kmeans",
                num_classes=8
            )
        }

        with open(
            os.path.join(self.metrics_dir, f"classification_epoch_{epoch}.json"),
            "w"
        ) as f:
            json.dump(cls_results, f, indent=4)

        # Convert pixel → image labels ONCE
        if y_val.ndim > 1:
            y_val_img = pixel_to_image_labels(y_val)
        else:
            y_val_img = y_val

        umap_emb, umap_labels = compute_umap_embeddings(reps_val, y_val_img)


        np.save(
            os.path.join(self.embeddings_dir, f"umap_epoch_{epoch}.npy"),
            umap_emb
        )
        def save_umap_plot(embeddings, labels, save_path, title=None):
            labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels

            plt.figure(figsize=(8, 8))
            scatter = plt.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=labels,
                cmap="tab20",
                s=6,
                alpha=0.8
            )

            if title:
                plt.title(title)

            plt.colorbar(scatter)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

                    
        save_umap_plot(
            umap_emb,
            umap_labels,
            save_path=os.path.join(self.plots_dir, f"umap_epoch_{epoch}.png"),
            title=f"UMAP – Epoch {epoch}"
        )


        seg_probe = train_linear_segmentation_probe(
            backbone=self.model,
            dataloader=train_loader,
            num_classes=8,
            feature_dim=self.config.ENCODER_DIM,
            epochs=20
        )

        seg_miou = evaluate_segmentation_probe(
            backbone=self.model,
            probe=seg_probe,
            dataloader=val_loader,
            num_classes=8
        )

        with open(
            os.path.join(self.metrics_dir, f"segmentation_epoch_{epoch}.json"),
            "w"
        ) as f:
            json.dump({"mIoU": seg_miou}, f, indent=4)
        # Store internally
        self.evaluation_results[epoch] = {
            "classification": cls_results,
            "segmentation": seg_miou
        }

    def train(self):
        print(f"Starting training for {self.config.EPOCHS} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Mask ratio: {self.config.MASK_RATIO}")
        print(f"{'='*80}")
        
        for epoch in range(self.current_epoch, self.config.EPOCHS):

            train_loss, train_contrast, train_mae = self.train_epoch(epoch)
            val_loss, val_contrast, val_mae = self.validate(epoch)
            
            print(f'\nEpoch {epoch} Summary:')
            print(f'Train - Loss: {train_loss:.4f}, Contrast: {train_contrast:.4f}, MAE: {train_mae:.4f}')
            print(f'Val   - Loss: {val_loss:.4f}, Contrast: {val_contrast:.4f}, MAE: {val_mae:.4f}')
            
            if epoch % 5 == 0:
                self.run_evaluation(epoch)
            
            # Save checkpoint
            if epoch % self.config.SAVE_FREQUENCY == 0 or epoch == self.config.EPOCHS - 1:
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_DIR,
                    f'checkpoint_epoch_{epoch}.pth'
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_loss,
                    path=checkpoint_path
                )
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth')
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    loss=val_loss,
                    path=best_path
                )
                print(f"Saved new best model with loss: {val_loss:.4f}")
            
            self.current_epoch = epoch + 1
        
        # Final evaluation
        print(f"\n{'='*80}")
        print("Training Complete! Running Final Evaluation...")
        print(f"{'='*80}")
        
        self.run_evaluation(self.config.EPOCHS - 1)
        
        # Save final model
        final_path = os.path.join(self.config.CHECKPOINT_DIR, 'final_model.pth')
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.config.EPOCHS,
            loss=self.best_loss,
            path=final_path
        )
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'evaluation_results': self.evaluation_results,
            'best_loss': self.best_loss,
            'config': {
                'epochs': self.config.EPOCHS,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'mask_ratio': self.config.MASK_RATIO,
            }
        }
        
        history_path = os.path.join(config.RESULTS_DIR, 'training_history.json')
        save_training_history(history, history_path)
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\n{'='*80}")
        print("Training Completed Successfully!")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"{'='*80}")


def main():
    # Set seed
    set_seed(42)
    
    # Load data
    print("Loading DFC data...")
    data_path = config.DATA_FILE
    train_images, train_labels, val_images, val_labels = load_dfc_data(data_path)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_images, train_labels, 
        val_images, val_labels,
        batch_size=config.BATCH_SIZE
    )
    
    # Create model
    print("Creating MARCo model...")
    model = MARCo(
        patch_size=config.PATCH_SIZE,
        encoder_dim=config.ENCODER_DIM,
        encoder_layers=config.ENCODER_LAYERS,
        attention_heads=config.ATTENTION_HEADS,
        decoder_dim=config.DECODER_DIM,
        decoder_layers=config.DECODER_LAYERS,
        total_channels=config.TOTAL_CHANNELS,
        num_patches=config.NUM_PATCHES
    )
    
    # Load checkpoint if specified
    if config.LOAD_CHECKPOINT:
        print(f"Loading checkpoint from {config.LOAD_CHECKPOINT}")
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, config.LOAD_CHECKPOINT, config.DEVICE
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        config=config,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels
    )
    
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":

    main()