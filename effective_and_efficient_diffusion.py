import argparse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import time

def generate_image(prompt, use_fp16=False, seed=0, scheduler_type="default", num_inference_steps=50):
    """
    Generate an image using Stable Diffusion with specified precision and scheduler.
    
    Args:
        prompt (str): The text prompt for image generation
        use_fp16 (bool): Whether to use FP16 precision (default: False)
        seed (int): Random seed for generation (default: 0)
        scheduler_type (str): Type of scheduler to use ('default' or 'DPM')
        num_inference_steps (int): Number of denoising steps (default: 50)
    Returns:
        tuple: (image, execution_time)
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
    
    # Set up generator with seed
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # Generate image
    image = pipeline(
        prompt,
        generator=generator,
        num_inference_steps=num_inference_steps
    ).images[0]
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    
    # Save image with appropriate suffix
    precision_suffix = "fp16" if use_fp16 else "fp32"
    scheduler_suffix = "_dpm" if scheduler_type.lower() == "dpm" else ""
    image.save(f"old_warrior_chief_{precision_suffix}{scheduler_suffix}.png")
    
    return image, execution_time

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
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("\nConfiguration:")
    print(f"Precision: {'FP16' if args.fp16 else 'FP32'}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Seed: {args.seed}")
    print(f"Prompt: {args.prompt}")
    print("\nGenerating image...")
    
    _, execution_time = generate_image(
        args.prompt,
        use_fp16=args.fp16,
        scheduler_type=args.scheduler,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed
    )
    print(f"Execution time: {execution_time:.2f} seconds")

