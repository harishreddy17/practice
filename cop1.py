from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# Load image captioning model
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load a pre-trained damage classifier (e.g., ResNet)
damage_classifier = models.resnet50(pretrained=True)
damage_classifier.fc = torch.nn.Linear(damage_classifier.fc.in_features, 2)  # 2 classes: damaged, not damaged
# Assuming you've fine-tuned this model; if not, you can train it separately on a damage dataset.

# Load fine-tuned weights (replace 'damage_classifier.pth' with your model path)
damage_classifier.load_state_dict(torch.load("damage_classifier.pth", map_location=torch.device("cpu")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)
damage_classifier.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Preprocessing for damage classifier
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_step(image_paths):
    captions = []
    damage_results = []
    
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        # Check for damage using the damage classifier
        classifier_input = classifier_transform(i_image).unsqueeze(0).to(device)
        with torch.no_grad():
            damage_logits = damage_classifier(classifier_input)
            damage_prediction = torch.argmax(damage_logits, dim=1).item()

        # Interpret damage results
        damage_status = "damaged" if damage_prediction == 1 else "not damaged"
        damage_results.append(damage_status)

        # Generate caption using image captioning model
        pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            output_ids = caption_model.generate(pixel_values, **gen_kwargs)

        caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        captions.append(caption)
    
    # Combine results
    final_results = [{"image": path, "caption": cap, "damage": damage}
                     for path, cap, damage in zip(image_paths, captions, damage_results)]
    return final_results

# Predict on a sample image
results = predict_step(['damaged.png']) 
for res in results:
    print(f"Image: {res['image']}, Caption: {res['caption']}, Damage: {res['damage']}")
