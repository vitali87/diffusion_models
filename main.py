from diffusers import DiffusionPipeline
from diffusers import EulerDiscreteScheduler

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", 
    use_safetensors=True,
    safety_checker=None
    ).to("cuda")

# Trying different scheduling methods
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

# Prompting the model
image = pipeline("An image of a squirrel in Picasso style").images[0]
image.save("image_of_squirrel_painting_euler.png")