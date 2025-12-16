#!/usr/bin/env python3
"""
Complete Training Script for Mistral Model
"""

import torch
from pinnacle_ai.core.models.mistral import MistralConfig, MistralForCausalLM
from pinnacle_ai.core.distributed import DistributedTrainer
from pinnacle_ai.core.optim import OptimizerBuilder, SchedulerBuilder
from pinnacle_ai.data import DataPipeline
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Mistral model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision")
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"], help="Distributed strategy")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large"], help="Model size")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model size configurations
    model_configs = {
        "small": MistralConfig(
            vocab_size=32000,
            hidden_size=2048,
            intermediate_size=7168,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=4,
        ),
        "medium": MistralConfig(
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
        ),
        "large": MistralConfig(
            vocab_size=32000,
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=64,
            num_attention_heads=64,
            num_key_value_heads=16,
        ),
    }
    
    config = model_configs[args.model_size]
    logger.info(f"Using {args.model_size} model configuration")
    
    # Initialize model
    model = MistralForCausalLM(config)
    logger.info(f"Model initialized: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
    
    # Initialize tokenizer (placeholder - would use actual tokenizer)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except ImportError:
        logger.warning("transformers not available. Using placeholder tokenizer.")
        tokenizer = None
    
    if tokenizer is None:
        raise RuntimeError("Tokenizer required. Install transformers: pip install transformers")
    
    # Initialize data pipeline
    data_pipeline = DataPipeline(
        data_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        distributed=args.distributed,
    )
    dataloader = data_pipeline.build()
    logger.info(f"Data pipeline initialized: {len(dataloader)} batches")
    
    # Initialize optimizer
    optimizer_builder = OptimizerBuilder(
        model=model,
        lr=args.learning_rate,
    )
    optimizer = optimizer_builder.build()
    logger.info("Optimizer initialized")
    
    # Initialize scheduler
    total_steps = len(dataloader) * args.num_epochs
    scheduler_builder = SchedulerBuilder(
        optimizer=optimizer,
        warmup_steps=1000,
        max_steps=total_steps,
    )
    scheduler = scheduler_builder.build()
    logger.info(f"Scheduler initialized: {total_steps} total steps")
    
    # Initialize distributed trainer
    trainer = None
    if args.distributed:
        trainer = DistributedTrainer(
            model=model,
            strategy=args.strategy,
            mixed_precision=args.mixed_precision,
        )
        trainer.set_optimizer(optimizer)
        logger.info(f"Distributed trainer initialized: {args.strategy}")
    else:
        # Non-distributed training
        if torch.cuda.is_available():
            model = model.cuda()
        if args.mixed_precision:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
        else:
            scaler = None
    
    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
        
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            if trainer:
                # Distributed training
                metrics = trainer.train_step(batch)
                loss = metrics["loss"]
            else:
                # Non-distributed training
                model.train()
                inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in batch.items()}
                
                if args.mixed_precision and scaler:
                    with autocast():
                        outputs = model(**inputs)
                        loss = outputs[1] if isinstance(outputs, tuple) else outputs
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(**inputs)
                    loss = outputs[1] if isinstance(outputs, tuple) else outputs
                    loss.backward()
                    optimizer.step()
                
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % 100 == 0:
                avg_loss = epoch_loss / (step + 1)
                logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
            
            if global_step % 1000 == 0:
                checkpoint_path = output_dir / f"checkpoint_{global_step}.pt"
                if trainer:
                    trainer.save_checkpoint(str(checkpoint_path))
                else:
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler else None,
                    }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            scheduler.step()
        
        logger.info(f"Epoch {epoch + 1} complete: avg_loss={epoch_loss/len(dataloader):.4f}")
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    if trainer:
        trainer.save_checkpoint(str(final_path))
    else:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
        }, final_path)
    logger.info(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()

