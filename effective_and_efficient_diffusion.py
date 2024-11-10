import argparse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from diffusers.utils import make_image_grid
import torch
import time
import os

def generate_image(prompts, use_fp16=False, seed=0, scheduler_type="default", num_inference_steps=50, output_dir="output_images", vae_id=None):
    """
    Generate images using Stable Diffusion with specified precision and scheduler.
    
    Args:
        prompts (str or list): Single prompt or list of prompts for image generation
        ...
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
    
    # Load custom VAE if specified
    if vae_id:
        vae = AutoencoderKL.from_pretrained(
            vae_id,
            torch_dtype=torch.float16 if use_fp16 else torch.float32
        )
        pipeline.vae = vae
        print(f"Loaded custom VAE: {vae_id}")
    
    # Configure scheduler if specified
    if scheduler_type.lower() == "dpm":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    
    pipeline = pipeline.to("cuda")
    
    # Convert single prompt to list if necessary
    if isinstance(prompts, str):
        prompts = [prompts]
    batch_size = len(prompts)
    
    # Set up generator with seed for batch processing
    generators = [torch.Generator("cuda").manual_seed(seed + i) for i in range(batch_size)]
    
    # Generate images
    output = pipeline(
        prompt=prompts,
        generator=generators,
        num_inference_steps=num_inference_steps
    )
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save image grid if batch_size > 1
    if batch_size > 1:
        # Calculate grid dimensions
        cols = min(batch_size, 3)  # Limit to 3 columns for better presentation
        rows = (batch_size + cols - 1) // cols
        
        # Pad the images list if necessary to match grid size
        total_cells = rows * cols
        padded_images = output.images + [output.images[-1]] * (total_cells - batch_size)
        
        grid_image = make_image_grid(padded_images, rows=rows, cols=cols)
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
    
    return output.images, execution_time

def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion with different configurations')
    parser.add_argument('--prompts', type=str, nargs='+', 
                      default=["portrait photo of a old warrior chief"],
                      help='One or more text prompts for image generation')
    parser.add_argument('--fp16', action='store_true',
                      help='Use FP16 precision instead of FP32')
    parser.add_argument('--scheduler', type=str, choices=['default', 'dpm'], default='default',
                      help='Type of scheduler to use')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                      help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed for generation')
    parser.add_argument('--output_dir', type=str, default="output_images",
                      help='Directory to save generated images')
    parser.add_argument('--vae', type=str, default=None,
                      help='HuggingFace model ID for custom VAE (e.g., stabilityai/sd-vae-ft-mse)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("\nConfiguration:")
    print(f"Precision: {'FP16' if args.fp16 else 'FP32'}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Seed: {args.seed}")
    print(f"Number of prompts: {len(args.prompts)}")
    print(f"Output directory: {args.output_dir}")
    print(f"VAE: {args.vae if args.vae else 'default'}")
    print("\nPrompts:")
    for i, prompt in enumerate(args.prompts, 1):
        print(f"{i}. {prompt}")
    print("\nGenerating images...")
    
    images, execution_time = generate_image(
        args.prompts,
        use_fp16=args.fp16,
        scheduler_type=args.scheduler,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        vae_id=args.vae
    )
    print(f"Generated {len(images)} images in {execution_time:.2f} seconds")

