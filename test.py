from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np
import matplotlib


def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask_2d = np.squeeze(mask) 
        
        mask_img = Image.fromarray(mask_2d)
        overlay = Image.new("RGBA", image.size, color + (0,))
        
        # Create alpha channel from the mask (50% opacity for the mask areas)
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        
        image = Image.alpha_composite(image, overlay)

    return image


# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load an image
image = Image.open("data/truck.jpg")
inference_state = processor.set_image(image)

# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="truck")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

overlayed_image = overlay_masks(image, masks)
overlayed_image.show()