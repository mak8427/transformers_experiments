from transformers import AutoImageProcessor, AutoModel
import torch

import matplotlib.pyplot as plt
import numpy as np

model_name = "facebook/dinov3-vitl16-pretrain-sat493m"  # smaller + sat-trained

processor = AutoImageProcessor.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_name).eval().to(device)

print("device:", device)

from PIL import Image
img = Image.open("/home/mak/Downloads/Sangano.png").convert("RGB")

inputs = processor(images=img, return_tensors="pt").to(device)
x = inputs['pixel_values']
print("input shape:", x.shape)

# take the tensor `x` that we gave to the model:
# shape is (1, 3, 224, 224)
# we remove the batch dimension [0], move channels to last position (H, W, C)
# and convert to a NumPy array for plotting
img_proc = x[0].cpu().permute(1, 2, 0).numpy()


# the processor normalized pixel values (roughly between -1 and +1)
# we shift and scale them back to 0–1 so matplotlib can display them
img_proc = (img_proc - img_proc.min()) / (img_proc.max() - img_proc.min())

# display the 224×224 image that the processor actually fed into the model
plt.imshow(img_proc)
plt.title("Image after processor (resized + normalized)")
plt.axis("off")                      # remove axis ticks for cleaner display
plt.show()