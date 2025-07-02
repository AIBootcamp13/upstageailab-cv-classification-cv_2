import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from model import get_model
import config

def inference():
    model = get_model(config.NUM_CLASSES)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    predictions = []
    for img_name in sorted(os.listdir(config.TEST_DIR)):
        img_path = f"{config.TEST_DIR}/{img_name}"
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        
        outputs = model(image)
        _, pred = outputs.max(1)
        
        predictions.append((img_name, pred.item()))
    
    submission = pd.DataFrame(predictions, columns=['ID', 'target'])
    submission.to_csv('submission.csv', index=False)
