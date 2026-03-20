"""
Monitor the AquaFormer training progress by reading TensorBoard event files.
Run this in a separate terminal:  python monitor_training.py
"""
import os
import sys
import time
import glob

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Install tensorboard first: pip install tensorboard")
    sys.exit(1)


def get_latest_metrics(log_dir='runs/aquaformer_v7'):
    """Read the latest scalar values from the most recent TensorBoard event file."""
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        return None
    
    # Sort by modification time and take the newest one
    latest_file = max(event_files, key=os.path.getmtime)
    
    ea = EventAccumulator(latest_file)
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    if not tags:
        return None

    metrics = {}
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            last = events[-1]
            metrics[tag] = {'step': last.step, 'value': last.value}
    return metrics


def print_progress(metrics):
    """Pretty-print the training progress."""
    if not metrics:
        print("  No metrics logged yet. Waiting ...")
        return

    epoch = 0
    for v in metrics.values():
        epoch = max(epoch, v['step'])

    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 60)
    print(f"  AquaFormer Training Monitor  |  Epoch: {epoch}/15")
    print("=" * 60)

    # Training
    print("\n  TRAINING")
    print("  " + "-" * 40)
    tr_loss = metrics.get('Loss/train', metrics.get('Loss/Train'))
    if tr_loss:
        print(f"    Train Loss:       {tr_loss['value']:.4f}")
    
    # Validation
    print("\n  VALIDATION")
    print("  " + "-" * 40)
    val_loss = metrics.get('Loss/val', metrics.get('Loss/Val'))
    if val_loss:
        print(f"    Val Loss:         {val_loss['value']:.4f}")
        
    full_r2 = metrics.get('Metrics/Val_FullR2', metrics.get('R2/RealSpace'))
    if full_r2:
        print(f"    Full R2 (Real):   {full_r2['value']:.4f}")
        
    rare_r2 = metrics.get('Metrics/Val_RareR2', metrics.get('R2/RareEvent99'))
    if rare_r2:
        print(f"    Rare Event R2:    {rare_r2['value']:.4f}")
        
    viol = metrics.get('Metrics/Val_ViolRate', metrics.get('Physics/ViolationRate'))
    if viol:
        print(f"    Violation Rate:   {viol['value']:.1f}%")
        
    log_r2 = metrics.get('R2/LogSpace') # Older runs
    if log_r2:
        print(f"    Log-Space R2:     {log_r2['value']:.4f}")
    
    cost = metrics.get('Cost/Score') # Older runs
    if cost:
        print(f"    Cost Score:       {cost['value']:.4f}")

    # Learning Rate
    if 'LR' in metrics:
        print(f"\n    Learning Rate:    {metrics['LR']['value']:.2e}")

    print("\n" + "=" * 60)
    print("  Refreshing every 30s ... Press Ctrl+C to stop.")


if __name__ == '__main__':
    log_dir = 'runs/aquaformer_v7'
    print(f"Monitoring: {log_dir}")
    print("Waiting for training to begin ...\n")

    while True:
        try:
            metrics = get_latest_metrics(log_dir)
            print_progress(metrics)
            time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"  Waiting for data ... ({e})")
            time.sleep(10)
