import torch
import PIL.Image
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # Add this before importing pyplot
import os

from diffusers import UNet2DModel
from diffusers import DDPMScheduler

torch.manual_seed(0)


repo_id = "google/ddpm-cat-256"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True).to("cuda")

scheduler = DDPMScheduler.from_pretrained(repo_id)

scheduler.set_timesteps(num_inference_steps)
print(f"Scheduler: {scheduler}")

print("### Important parameters of the scheduler ###")
print(f"Number of training steps: {scheduler.config.num_train_timesteps}")
print(f"Number of inference steps: {len(scheduler.timesteps)}")

print("### Important parameters of the model ###")
print(f"Sample size: {model.config.sample_size}")
print(f"In channels: {model.config.in_channels}")
print(f"Down block types: {model.config.down_block_types}")
print(f"Up block types: {model.config.up_block_types}")
print(f"Block out channels: {model.config.block_out_channels}")
print(f"Layers per block: {model.config.layers_per_block}")

noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size).to("cuda")


with torch.no_grad():
    noisy_residuals = model(sample=noisy_sample, timestep=2).sample

print("### Noisy sample and residuals ###")
print(f"Noisy sample shape: {noisy_sample.shape}")
print(f"Noisy residuals shape: {noisy_residuals.shape}")

def save_sample(sample, i):
    # Create output directory if it doesn't exist
    os.makedirs('output_images', exist_ok=True)
    
    # Process the image
    image_processed = sample.cpu().permute(0, 2, 3, 1)[0]  # Remove batch dimension and get correct format
    image_processed = (image_processed * 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)
    
    # Create and save the plot
    plt.figure(figsize=(10, 10))
    plt.imshow(image_processed)
    plt.axis('off')
    plt.title(f'Step {i}')
    plt.savefig(f'output_images/step_{i:04d}.png')
    plt.close()  # Important to free memory


sample = noisy_sample

for i,t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict the noise residual
    with torch.no_grad():
        residual = model(sample, t).sample
  
    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample
    
    # 3. visualize the image
    if (i+1) % 50 == 0:
        save_sample(sample, i+1)


