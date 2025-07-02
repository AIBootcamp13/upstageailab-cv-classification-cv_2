import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import torch.optim as optim
from dataset import DocumentDataset
from model import get_model

import wandb
import config
from torchvision import transforms

def train():
    wandb.init(project="document-classification", entity="boot_camp_13_2nd_group_2nd")

    # wandb.config로 하이퍼파라미터도 기록 가능
    wandb.config.update({
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE
    })

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    dataset = DocumentDataset(config.TRAIN_CSV, config.TRAIN_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config.NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        wandb.log({"epoch": epoch+1, "loss": avg_loss})

    torch.save(model.state_dict(), 'model.pth')
