import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.goal_dataset import GoalDataset
from src.models.goal_detection import GoalDetector
from torchvision import transforms

def train_goal_detector(train_dir='data/goals/train', val_dir='data/goals/val', num_epochs=50, batch_size=16, learning_rate=0.001, save_path='models/goal_detector.pth'):
    """
    Train the goal detection model.

    Parameters:
    - train_dir (str): Path to the training dataset.
    - val_dir (str): Path to the validation dataset.
    - num_epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - learning_rate (float): Learning rate for the optimizer.
    - save_path (str): Path to save the trained model.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = GoalDataset(root_dir=train_dir, transform=transform)
    val_dataset = GoalDataset(root_dir=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GoalDetector(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    torch.save(model.state_dict(), save_path)