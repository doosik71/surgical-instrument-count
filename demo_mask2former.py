import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from transformers import Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt

model_name = "facebook/mask2former-swin-large-coco-instance"

# load Mask2Former fine-tuned on COCO instance segmentation
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)

image = Image.open("./data/truck.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

result = processor.post_process_instance_segmentation(
    outputs, target_sizes=[image.size[::-1]], threshold=0.2)[0]

predicted_instance_map = result["segmentation"]

print(torch.unique(predicted_instance_map))

# Visualize the result
plt.imshow(image)
# Use a colormap to visualize the instance map
# The 'jet' colormap is a good choice for this
# We need to create a masked array to overlay the segmentation
# where the background (instance -1) is transparent.
instance_map_np = predicted_instance_map.numpy()
masked_instance_map = np.ma.masked_where(
    instance_map_np == -1, instance_map_np)

plt.imshow(masked_instance_map, cmap='jet', alpha=0.5)
plt.axis('off')
plt.show()
