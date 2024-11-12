
import torch 
import matplotlib.pyplot as plt  
from src.models.unet import UNet, conv_block, upsampling_block 
from src.data.data_processing import get_dataloader
from src.training.train_model import train_model

def display(display_list):
    plt.figure(figsize=(15, 15))
    titles = ['Input Image', 'True Mask', 'Predicted Mask']

    for i, item in enumerate(display_list):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])

        if isinstance(item, torch.Tensor):
            item = item.cpu().detach()  # pushed to CPU, detached from the computational graph


        if item.ndim == 3:      # [C, H, W] -->[H, W, C] for images
            item = item.permute(1, 2, 0)
            plt.imshow(item)
        elif item.ndim == 2:
            plt.imshow(item, cmap='tab20')

        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    pred_mask = torch.argmax(pred_mask, dim=1).detach()
    pred_mask = pred_mask.cpu()
    pred_mask = pred_mask.to(torch.uint8)
    return pred_mask

def show_predictions(model,dataloader, device, num=1):

    model.eval()
    for i, batch in enumerate(dataloader):

        if batch["IMAGE"].size(0) < 1: 
            continue

        IMG = batch["IMAGE"][0].to(device).unsqueeze(0)  # Shape-->[1, C, H, W]
        MASK = batch["MASK"][0].to(device)  # Shape--> [H, W]

        with torch.no_grad():
            pred_mask = model(IMG)

        input_img = IMG[0].cpu() / 255.0 if IMG.max() > 1 else IMG[0].cpu()
        true_mask = MASK.cpu()
        predicted_mask = create_mask(pred_mask).cpu()

        display([input_img, true_mask, predicted_mask])

        if i >= num - 1:
            break
