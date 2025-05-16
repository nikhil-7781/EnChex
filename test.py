import os
import torch
import argparse
import numpy as np
from models import build_model
from data import build_loader
from config import get_config
from xai_utils.explainability import GradCAM

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Config file path')
    parser.add_argument('--testset', required=True, help='Path to test dataset')
    parser.add_argument('--test_csv_path', required=True, help='Path to test CSV')
    parser.add_argument('--model_path', required=True, help='Path to trained model .pth file')
    parser.add_argument('--output_dir', default='test_results', help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    config = get_config(args)
    return args, config

def main():
    args, config = parse_option()
    
    # Build test loader
    _, _, _, _, _, data_loader_test, _ = build_loader(
        config,
        test_dir=args.testset,
        test_csv=args.test_csv_path
    )
    
    # Build model
    model = build_model(config)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda().eval()  # Keep model in eval mode
    
    # Create Grad-CAM with proper target layer (adjust based on your architecture)
    # Example: Use last CNN layer from hybrid model
    target_layer = model.cnn.layer4[-1] if hasattr(model, 'cnn') else model.swin.layers[-1].blocks[-1].norm1
    grad_cam = GradCAM(model, target_layer)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run test set with XAI visualizations
    print("Generating XAI visualizations...")
    for batch_idx, (images, _) in enumerate(data_loader_test):
        images = images.cuda()
        
        # Critical fix: enable gradients for Grad-CAM
        with torch.enable_grad():
            # Forward pass with gradient tracking
            outputs = model(images)
            logits = outputs['logits']
            target_class = torch.argmax(logits, dim=1)
            
            # Generate CAM
            cam_dict = grad_cam.generate_cam(images, target_class=target_class)
        
        # Save visualization
        grad_cam.visualize(
            cam_dict,
            save_path=os.path.join(args.output_dir, f'test_sample_{batch_idx}.png')
        )
        
        if batch_idx >= args.num_samples - 1:
            break

    print(f"Saved {args.num_samples} visualizations to {args.output_dir}")

if __name__ == '__main__':
    main()
