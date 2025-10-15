import typer
import yaml
from dataloader import SegDataModule

from trainers.pl_trainer import SRSegLit, build_trainer
from trainers.unetpp_trainer import BuildingTrainer, build_unetpp_trainer
from trainers.dlinknet_trainer import RoadTrainer, build_dlinknet_trainer

def get_trainer_type(cfg: dict) -> str:
    """
    Determine the trainer type based on configuration.
    
    Args:
        cfg (dict): Configuration dictionary
        
    Returns:
        str: Trainer type ('multi_task', 'road_only', 'building_only')
    """
    task_cfg = cfg.get('task', {})
    
    # Handle both string and dict formats
    if isinstance(task_cfg, str):
        task_type = task_cfg.lower()
    elif isinstance(task_cfg, dict) and 'type' in task_cfg:
        task_type = task_cfg['type'].lower()
    else:
        # Default fallback
        return 'multi_task'
    
    # Check task type
    if task_type in ['road', 'roads', 'road_only']:
        return 'road_only'
    elif task_type in ['building', 'buildings', 'building_only']:
        return 'building_only'
    elif task_type in ['multi_task', 'both', 'multitask']:
        return 'multi_task'
    else:
        # Default fallback
        return 'multi_task'

# ...existing code...
def main(config: str = typer.Option('configs/config.yaml', help='Path to YAML configuration file')):
    """
    Train a segmentation model with automatic trainer selection.
    Supports multi-task (BR Collaborative), road-only (DLinkNet), or building-only (UNet++) training.
    """
    # Load configuration
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Determine trainer type
    trainer_type = get_trainer_type(cfg)
    
    print(f"🚀 Detected trainer type: {trainer_type}")
    print(f"📁 Using config: {config}")

    # Initialize DataModule
    data_module = SegDataModule(cfg)
    data_module.setup()

    # ──────────────────────────────────────────────────────────────────────────
    # Build model and trainer based on detected type
    if trainer_type == 'multi_task':
        print("🏗️  Using multi-task training (BR Collaborative)")      
        model = SRSegLit(cfg)
        trainer = build_trainer(cfg)
        
    elif trainer_type == 'road_only':
        print("🛣️  Using road-only training (DLinkNet)")              
        model = RoadTrainer(cfg)
        trainer = build_dlinknet_trainer(cfg)
        
    elif trainer_type == 'building_only':
        print("🏢 Using building-only training (UNet++)")        
        model = BuildingTrainer(cfg)
        trainer = build_unetpp_trainer(cfg)
        
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

    # ──────────────────────────────────────────────────────────────────────────
    # Display training info
    print("\n" + "="*50)
    print(f"🎯 Task: {trainer_type.replace('_', ' ').title()}")
    print(f"🔧 Model: {model.__class__.__name__}")
    print(f"📊 Max Epochs: {cfg.get('training', {}).get('epochs', 'Unknown')}")
    print(f"📈 Learning Rate: {cfg.get('training', {}).get('lr', 'Unknown')}")
    print(f"🎮 Device: {'GPU' if trainer.accelerator == 'gpu' else 'CPU'}")
    print("="*50 + "\n")

    # Train
    trainer.fit(model, data_module)
    print("\nTraining complete! 🎉")


if __name__ == '__main__':
    typer.run(main)

#    nohup python main.py --config configs/dlink_config.yaml > logs/road_train.log 2>&1 &
#    nohup python main.py --config configs/unetpp_config.yaml > logs/building_train.log 2>&1 &
#    nohup python main.py --config configs/br_config.yaml > logs/multi_task_train.log 2>&1 &