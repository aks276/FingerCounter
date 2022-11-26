import torch
from torch import nn as nn
from tqdm import tqdm

import os

from data import train_dataloader, test_dataloader
# from arch import AlexNet
from torchvision.models import vgg16
import config

# model = AlexNet()
model = vgg16()
model.classifier[6].out_features = 6

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=config.LEARNING_RATE)

model = model.to(config.DEVICE)

for epoch in range(config.EPOCHS):
    running_loss = 0
    for img, cls in tqdm(train_dataloader):
        img = img.to(config.DEVICE)
        cls = cls.to(config.DEVICE)
        model.train()
        
        op = model(img)

        loss = loss_fn(op, cls)
        running_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Train Loss : {running_loss / 1125}")

    test_loss = 0
    model.eval()
    with torch.no_grad():
        for img, cls in tqdm(test_dataloader):
            img = img.to(config.DEVICE)
            cls = cls.to(config.DEVICE)


            op = model(img)
            loss = loss_fn(op, cls)
            test_loss += loss
    print(f"Test Loss: {test_loss / len(train_dataloader)}")
    
    torch.save(model.state_dict(), os.path.join(config.MODEL_PATH, f'model_{epoch}.pth'))