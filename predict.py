import torch
from torchvision import transforms, models
from PIL import Image

# Define correct class-to-label mapping (alphabetically sorted as ImageFolder does)
disease_classes = {
    0: ("Mossaic Virus", "Use virus-resistant varieties and control insect vectors.", "https://www.youtube.com/embed/XXXXX"),
    1: ("Southern blight", "Apply fungicides like PCNB, and improve soil drainage.", "https://www.youtube.com/embed/XXXXX"),
    2: ("Sudden Death Syndrome", "Use resistant varieties and rotate crops.", "https://www.youtube.com/embed/XXXXX"),
    3: ("Yellow Mosaic", "Control whiteflies and use virus-free seeds.", "https://www.youtube.com/embed/XXXXX"),
    4: ("bacterial_blight", "Use copper-based bactericides and remove infected leaves.", "https://www.youtube.com/embed/6kJ_0Y8y3XU"),
    5: ("brown_spot", "Use fungicides like chlorothalonil and improve air flow.", "https://www.youtube.com/embed/XXXXX"),
    6: ("crestamento", "Ensure proper nutrient balance and avoid water stress.", "https://www.youtube.com/embed/XXXXX"),
    7: ("ferrugen", "Improve soil health and investigate for nutrient issues.", "https://www.youtube.com/embed/XXXXX"),
    8: ("powdery_mildew", "Use sulfur-based fungicides and increase spacing.", "https://www.youtube.com/embed/XXXXX"),
    9: ("septoria", "Remove infected debris and apply protective fungicides.", "https://www.youtube.com/embed/XXXXX")
}

# Define transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model
def load_model(model_path='model/soybean_model.pt'):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(disease_classes))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Predict disease
def predict_disease(image_path, model):
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        label_index = predicted.item()
    
    print("Predicted index:", label_index)  # Debug line
    
    disease, solution, video_url = disease_classes.get(label_index, ("Unknown", "No solution available.", ""))
    return {
        "Disease": disease,
        "Recommended Solution": solution,
        "Video": video_url
    }
