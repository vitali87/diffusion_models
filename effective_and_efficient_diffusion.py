import argparse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import time
from diffusers.utils import make_image_grid
import os

def generate_image(prompt, use_fp16=False, seed=0, scheduler_type="default", num_inference_steps=50, batch_size=1, output_dir="output_images"):
    """
    Generate an image using Stable Diffusion with specified precision and scheduler.
    
    Args:
        prompt (str): The text prompt for image generation
        use_fp16 (bool): Whether to use FP16 precision (default: False)
        seed (int): Random seed for generation (default: 0)
        scheduler_type (str): Type of scheduler to use ('default' or 'DPM')
        num_inference_steps (int): Number of denoising steps (default: 50)
        batch_size (int): Number of images to generate in parallel (default: 1)
        output_dir (str): Directory to save generated images (default: 'output_images')
    Returns:
        tuple: (images, execution_time)
    """
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    
    start_time = time.time()
    
    # Set up model configuration based on precision
    model_kwargs = {
        "use_safetensors": True,
        "torch_dtype": torch.float16 if use_fp16 else torch.float32
    }
    
    # Initialize pipeline with appropriate settings
    pipeline = DiffusionPipeline.from_pretrained(model_id, **model_kwargs)
    
    # Configure scheduler if specified
    if scheduler_type.lower() == "dpm":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    
    pipeline = pipeline.to("cuda")
    
    # Set up generator with seed for batch processing
    generators = [torch.Generator("cuda").manual_seed(seed + i) for i in range(batch_size)]
    prompts = [prompt] * batch_size
    
    # Generate images
    output = pipeline(
        prompts,
        generator=generators,
        num_inference_steps=num_inference_steps
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save image grid if batch_size > 1
    if batch_size > 1:
        # Calculate grid dimensions (trying to make it as square as possible)
        grid_size = int(batch_size ** 0.5)
        rows = grid_size
        cols = (batch_size + grid_size - 1) // grid_size  # Ceiling division
        
        # Create and save grid
        grid_image = make_image_grid(output.images, rows=rows, cols=cols)
        precision_suffix = "fp16" if use_fp16 else "fp32"
        scheduler_suffix = "_dpm" if scheduler_type.lower() == "dpm" else ""
        grid_filename = os.path.join(output_dir, f"old_warrior_chief_{precision_suffix}{scheduler_suffix}_grid.png")
        grid_image.save(grid_filename)
        print(f"Saved grid image as: {grid_filename}")
    
    # Save individual images
    for idx, image in enumerate(output.images):
        precision_suffix = "fp16" if use_fp16 else "fp32"
        scheduler_suffix = "_dpm" if scheduler_type.lower() == "dpm" else ""
        batch_suffix = f"_batch{idx}"
        filename = os.path.join(output_dir, f"old_warrior_chief_{precision_suffix}{scheduler_suffix}{batch_suffix}.png")
        image.save(filename)
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    
    return output.images, execution_time

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion with different configurations')
    parser.add_argument('--prompt', type=str, default="portrait photo of a old warrior chief",
                      help='Text prompt for image generation')
    parser.add_argument('--fp16', action='store_true',
                      help='Use FP16 precision instead of FP32')
    parser.add_argument('--scheduler', type=str, choices=['default', 'dpm'], default='default',
                      help='Type of scheduler to use')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                      help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed for generation')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Number of images to generate in parallel')
    parser.add_argument('--output_dir', type=str, default="output_images",
                      help='Directory to save generated images')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("\nConfiguration:")
    print(f"Precision: {'FP16' if args.fp16 else 'FP32'}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Seed: {args.seed}")
    print(f"Prompt: {args.prompt}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print("\nGenerating images...")
    
    images, execution_time = generate_image(
        args.prompt,
        use_fp16=args.fp16,
        scheduler_type=args.scheduler,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    print(f"Generated {len(images)} images in {execution_time:.2f} seconds")

