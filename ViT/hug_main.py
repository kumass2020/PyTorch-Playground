from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

for i in range(100):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][i]

    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])