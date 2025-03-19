import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class GoalDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(GoalDetector, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

def load_goal_detector(model_path='models/goal_detector.pth'):
    """
    Load the trained goal detection model.

    Parameters:
    - model_path (str): Path to the model weights.

    Returns:
    - model (GoalDetector): Loaded model.
    """
    model = GoalDetector(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def detect_goal(frame, model, transform):
    """
    Detect goal events in a frame using a pre-trained model.

    Parameters:
    - frame (np.ndarray): Input frame.
    - model (GoalDetector): Trained goal detection model.
    - transform (transforms.Compose): Transformation pipeline for the frame.

    Returns:
    - goal_detected (bool): True if a goal is detected, False otherwise.
    """
    image = Image.fromarray(frame)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item() == 1