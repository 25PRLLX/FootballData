import torch
from torch.utils.data import DataLoader
from src.data.goal_dataset import GoalDataset
from src.models.goal_detection import GoalDetector
from torchvision import transforms

def evaluate_goal_detector(model_path='models/goal_detector.pth', val_dir='data/goals/val'):
    """
    Evaluate the goal detection model.

    Parameters:
    - model_path (str): Path to the model weights.
    - val_dir (str): Path to the validation dataset.
    """
    model = GoalDetector(num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = GoalDataset(root_dir=val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')