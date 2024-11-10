from diffusers import UNet2DModel

repo_id = "google/ddpm-cat-256"
model = UNet2DModel.from_pretrained(repo_id, use_safetensors=True)

print(f"Sample size: {model.config.sample_size}")
print(f"In channels: {model.config.in_channels}")
print(f"Down block types: {model.config.down_block_types}")
print(f"Up block types: {model.config.up_block_types}")
print(f"Block out channels: {model.config.block_out_channels}")
print(f"Layers per block: {model.config.layers_per_block}")