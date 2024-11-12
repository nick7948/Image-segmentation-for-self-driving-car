import os
import cv2
import matplotlib.pyplot as plt

def plot_images_and_masks(data_folder, num_images_to_plot=30):

    image_subfolder_path = os.path.join(data_folder, 'train')
    mask_subfolder_path = os.path.join(data_folder, 'gtFine_trainvaltest (2)', 'gtFine', 'train')


    for city_folder in os.listdir(image_subfolder_path):
        city_image_path = os.path.join(image_subfolder_path, city_folder)
        city_mask_path = os.path.join(mask_subfolder_path, city_folder)

        image_files = sorted(os.listdir(city_image_path))
        mask_files = sorted(os.listdir(city_mask_path))


        mask_files = [f for f in mask_files if 'labelIds' in f]

        for i in range(min(num_images_to_plot, len(image_files), len(mask_files))):
            image_path = os.path.join(city_image_path, image_files[i])
            mask_path = os.path.join(city_mask_path, mask_files[i])

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'Image {i+1} - City: {city_folder}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title(f'Mask {i+1} - City: {city_folder}')
            plt.axis('off')

            plt.show()

