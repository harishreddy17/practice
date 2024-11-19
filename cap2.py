import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import time

# Load the ViT-GPT2 model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Setup device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Video and output paths
video_path = 'Test_Videos/demo.mp4'
output_path = 'output_video.mp4'

# Video capture and writer
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Object classes of interest
object_classes = ['person', 'chair', 'bottle', 'laptop', 'car']

# Generate caption for the frame
def generate_caption(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pixel_values = feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)

    # Create attention mask (no padding needed, but let's create one for consistency)
    attention_mask = torch.ones(pixel_values.shape, device=device)

    with torch.no_grad():  # Disable gradient computation for inference
        output_ids = model.generate(pixel_values, attention_mask=attention_mask, max_length=16, num_beams=4)
    
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    print(f"Generated Caption: {caption}")  # Debugging caption
    return caption

# Extract objects from the caption
def extract_objects_from_caption(caption):
    objects = []
    for obj in object_classes:
        if obj in caption:
            objects.append(obj)
    return objects

# Draw results on the frame
def draw_results(frame, caption, objects):
    # Draw the caption at the top of the frame
    cv2.putText(frame, f"Caption: {caption}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for idx, obj in enumerate(objects):
        cv2.putText(frame, f"Object {idx+1}: {obj}", (10, 60 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Generate caption and extract objects
    caption = generate_caption(frame)
    objects = extract_objects_from_caption(caption)

    # Draw the caption and object labels on the frame
    draw_results(frame, caption, objects)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow("Captioning and Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
