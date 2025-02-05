import os
import torch
import gradio as gr
from torch import nn
from torchvision import models, transforms
from PIL import Image


class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.model = models.squeezenet1_1(weights=None)

        # Get the number of input features from the last conv layer
        in_features = self.model.classifier[1].in_channels

        # Replace the classifier with a new sequential block
        self.model.classifier = nn.Sequential(
            nn.Conv2d(in_features, 512, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def load_model(model_path: str, device: torch.device) -> nn.Module:
    """Load the pre-trained model from the checkpoint."""
    model = AgeClassifier()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess the input image for prediction."""

    # Get model-specific transform functions
    weights = models.EfficientNet_V2_M_Weights.DEFAULT
    transformation = weights.transforms()

    image = Image.open(image_path).convert("RGB")
    image = transformation(image).unsqueeze(0)  # Add batch dimension
    return image


def predict(image_path: str, model: nn.Module, device: torch.device) -> str:
    """Make a prediction about the person's age."""
    image = preprocess_image(image_path)
    image = image.to(device)

    outputs = model(image)
    preds = (outputs > 0.5).float()

    if preds.item() == 1:
        return "The person on the picture is 18 years old or older."
    else:
        return "The person on the picture is under 18 years old."


def process_image(image) -> str:
    """Return the filename of the uploaded image."""
    if image is None:
        return "No file uploaded"
    filename = os.path.basename(image)
    return f"Filename: {filename}"


# Initialize the device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoint_best_val_los.pth"
model = load_model(model_path, device)


with gr.Blocks() as demo:
    gr.Markdown("## Upload an Image")

    with gr.Row():
        image_input = gr.Image(type="filepath", height=400, width=400)
        filename_output = gr.Textbox(label="Age classification result")

    image_input.change(
        lambda image: predict(image, model, device),
        inputs=image_input,
        outputs=filename_output,
    )

demo.launch()
