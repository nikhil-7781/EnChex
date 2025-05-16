import os
import torch
import argparse
from models import build_model
from data import build_loader
from config import get_config
from xai_utils.explainability import GradCAM

def parse_option():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--cfg', required=True, help='Config file path')
    parser.add_argument('--testset', required=True, help='Path to test dataset')
    parser.add_argument('--test_csv_path', required=True, help='Path to test CSV')  # <-- FIXED
    parser.add_argument('--model_path', required=True, help='Path to trained model .pth')
    parser.add_argument('--output_dir', required=True, help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')

    # Dummy arguments for config compatibility
    parser.add_argument("--local_rank", type=int, default=0, help="Dummy - not used in testing")
    parser.add_argument('--output', default='', help='Dummy - not used')
    parser.add_argument('--data-path', default='', help='Dummy - not used')
    parser.add_argument('--zip', action='store_true', help='Dummy - not used')
    parser.add_argument('--cache-mode', default='part', help='Dummy - not used')
    parser.add_argument('--resume', default='', help='Dummy - not used')
    parser.add_argument('--accumulation-steps', type=int, default=0, help='Dummy - not used')
    parser.add_argument('--use-checkpoint', action='store_true', help='Dummy - not used')
    parser.add_argument('--amp-opt-level', default='O1', help='Dummy - not used')
    parser.add_argument('--tag', default='', help='Dummy - not used')
    parser.add_argument('--eval', action='store_true', help='Dummy - not used')
    parser.add_argument('--throughput', action='store_true', help='Dummy - not used')
    parser.add_argument('--batch-size', type=int, help='Dummy - uses config value')
    parser.add_argument('--opts', nargs='+', default=None)
    parser.add_argument('--trainset', default='')
    parser.add_argument('--validset', default='')
    parser.add_argument('--train_csv_path', default='')
    parser.add_argument('--valid_csv_path', default='')
    parser.add_argument('--num_mlp_heads', type=int, default=3)

    args = parser.parse_args()
    config = get_config(args)
    return args, config



def main():
    args, config = parse_option()
    
    # Build test loader
    _, _, _, _, _, data_loader_test, _ = build_loader(
        config,
        test_dir=args.testset,
        test_csv=args.test_csv
    )
    
    # Build model
    model = build_model(config)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda().eval()
    
    # Create Grad-CAM - adjust target layer for your architecture
    target_layer = model.cnn.layer4[-1] if hasattr(model, 'cnn') else model.swin.layers[-1].blocks[-1].norm1
    grad_cam = GradCAM(model, target_layer)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run test set with XAI visualizations
    print(f"Generating XAI visualizations for {args.num_samples} samples...")
    for batch_idx, (images, _) in enumerate(data_loader_test):
        images = images.cuda()
        with torch.enable_grad():  # Critical fix for gradients
            outputs = model(images)
            logits = outputs['logits']
            target_class = torch.argmax(logits, dim=1)
            cam_dict = grad_cam.generate_cam(images, target_class=target_class)
        
        grad_cam.visualize(
            cam_dict,
            save_path=os.path.join(args.output_dir, f'sample_{batch_idx}.png')
        )
        
        if batch_idx >= args.num_samples - 1:
            break

    print(f"Saved {args.num_samples} visualizations to:\n{args.output_dir}")

if __name__ == '__main__':
    main()
