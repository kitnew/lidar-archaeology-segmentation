import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import wandb
import random
import numpy as np

# Use absolute imports for testing local packages
from engine.trainer import Trainer

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    log.info(f"Instantiating training pipeline with config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device('cuda')
    log.info(f"Using device: {device}")

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize Logger
    logger = None
    if cfg.logger.use_wandb:
        wandb.init(
            project=cfg.logger.project,
            name=cfg.logger.name,
            group=cfg.logger.group,
            tags=list(cfg.logger.tags),
            config=OmegaConf.to_container(cfg, resolve=True)  # pyright: ignore[reportArgumentType]
        )
        logger = wandb
        log.info("Initialized WandB logger.")

    # 1. Instantiate Model
    log.info("Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    model.to(device)

    # 2. Instantiate Data
    log.info("Instantiating dataset...")
    
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    val_dataset = hydra.utils.instantiate(cfg.dataset.val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    log.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Instantiate Loss and Optimizer
    log.info("Instantiating loss & optimizer...")
    criterion = hydra.utils.instantiate(cfg.loss).to(device)
    log.info(f"Using loss: {criterion.__class__.__name__}")
    # The optimizer needs model parameters, so we can't fully instantiate it from cfg alone easily in one step if it's strict.
    # Hydra provides a nice way: instantiate a partial using _partial_: true OR just pass the parameters explicitly.
    # For prototype simplicity:
    # optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    log.info(f"Using optimizer: {optimizer.__class__.__name__}")

    # 4. Instantiate and run Trainer
    log.info("Starting training loop...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        threshold=cfg.threshold,
        logger=logger,
        checkpoint_metric=cfg.checkpoint_metric,
    )
    
    # hydra.core.hydra_config.HydraConfig.get().runtime.output_dir get's the running output folder
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # pyright: ignore[reportAttributeAccessIssue]
    log.info(f"Saving checkpoints to {output_dir}")
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        save_dir=output_dir
    )
    
    log.info("Training complete")
    if logger:
        wandb.finish()

if __name__ == "__main__":
    main()
