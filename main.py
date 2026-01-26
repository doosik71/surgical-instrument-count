import torch
from huggingface_hub import login
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from PIL import Image
# from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


login(new_session=False)

pipe = pipeline("mask-generation", model="facebook/sam3")
tokenizer = AutoTokenizer.from_pretrained("facebook/sam3")
model = AutoModel.from_pretrained("facebook/sam3")
processor = Sam3Processor(model)

image = Image.open("data/1769044153248.jpg")
inference_state = processor.set_image(image)
output = processor.set_text_prompt(
    state=inference_state, prompt="Surgical instruments")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print(masks, boxes, scores)
