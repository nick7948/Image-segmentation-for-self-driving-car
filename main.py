
import torch
import torch.optim as optim
import torch.nn as nn
import os


from src.data.data_processing import get_dataloader
from src.models.unet import UNet
from src.training.train_model import train_model
from src.model_prediction import show_predictions

def main():
    
    image_dir = "/content/drive/MyDrive/train"  
    mask_dir = "/content/drive/MyDrive/gtFine_trainvaltest (2)/gtFine/train"  
    if not os.path.exists(image_dir):
        print(f"Error: Image path '{image_dir}' not found.")
        return
    batch_size = 4
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(image_dir, mask_dir, batch_size=batch_size)

    model = UNet().to(device)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    train_model(model, dataloader, criterion, optimizer,device, num_epochs=num_epochs)

    show_predictions(model, dataloader, device, num=3) 
if __name__ == "__main__":
    main()

