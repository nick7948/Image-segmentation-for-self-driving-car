
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import sys
sys.path.append('/content/Image-segmentation-for-self-driving-car/src')

from models.unet import UNet
from data.data_processing import get_dataloader

def train_model(model, dataloader, criterion, optimizer, device, num_epochs):
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()  
        epoch_losses = []  

        for i, batch in enumerate(dataloader):
            
            images = batch['IMAGE'].to(device) 
            masks = batch['MASK'].to(device)
            images = images.type(torch.float32) / 255.0 #UNT

            
            if masks.dim() == 4 and masks.shape[1] == 1:  # actual--> [N, 1, H, W]
                masks = masks.squeeze(1)  # con--> [N, H, W]

            masks = masks.long()  

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)
            epoch_losses.append(loss.item() * images.size(0))  

            loss.backward()
            optimizer.step()

            
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        
        avg_epoch_loss = np.sum(epoch_losses) / len(dataloader.dataset)
        losses.append(avg_epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}] completed with average loss: {avg_epoch_loss:.4f}')
