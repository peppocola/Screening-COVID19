import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torchvision.utils

MISS_IMAGES_PATH = 'misses'


def save_history(history, filepath):
    fig, axs = plt.subplots(2, tight_layout=True)
    axs[0].set_title('loss')
    axs[0].plot(history['train']['loss'], label='train')
    axs[0].plot(history['validation']['loss'], label='validation')
    axs[1].set_title('accuracy')
    axs[1].plot(history['train']['accuracy'], label='train')
    axs[1].plot(history['validation']['accuracy'], label='validation')
    axs[0].legend()
    axs[1].legend()
    fig.savefig(filepath, dpi=192)


def plot_cxr2_errors(dataset, errors, model_name):
    label_names = ['negative', 'positive']

    miss_images_path = os.path.join('cxr2-' + MISS_IMAGES_PATH, model_name)
    if not os.path.isdir(miss_images_path):
        os.mkdir(miss_images_path)

    for (idx, label) in errors:
        tensor, _ = dataset.__getitem__(idx)
        img = tensor.numpy().squeeze()
        plt.figure(figsize=(4, 4))
        plt.title(label_names[label])
        path = os.path.join(miss_images_path, 'predict_error{}_{}.png'.format(idx, label_names[label]))
        plt.imsave(path, img, cmap='gray', format='png')
        plt.close()


def save_attention_map(filepath, img, att1, att2):
    # Move on CPU
    img = img.cpu()
    att1 = att1.cpu()
    att2 = att2.cpu()

    # Un-normalize attention maps
    att1 = (att1 - att1.min()) / (att1.max() - att1.min()) * 255
    att2 = (att2 - att2.min()) / (att2.max() - att2.min()) * 255
    img = img * 255.0

    # Upsample attention maps
    att1 = torch.nn.functional.interpolate(att1, scale_factor=(8, 8), mode='bilinear', align_corners=True)
    att2 = torch.nn.functional.interpolate(att2, scale_factor=(16, 16), mode='bilinear', align_corners=True)

    # Conver to Numpy arrays
    att1 = att1.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    att2 = att2.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    img = img.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

    # Apply colormap
    att1 = cv2.applyColorMap(att1, cv2.COLORMAP_JET)
    att2 = cv2.applyColorMap(att2, cv2.COLORMAP_JET)

    # Combine heatmaps
    img = np.repeat(img, repeats=3, axis=2)
    img_att1 = cv2.addWeighted(img, 0.7, att1, 0.3, 0.0)
    img_att2 = cv2.addWeighted(img, 0.7, att2, 0.3, 0.0)
    img_att1 = cv2.cvtColor(img_att1, cv2.COLOR_BGR2RGB)
    img_att2 = cv2.cvtColor(img_att2, cv2.COLOR_BGR2RGB)

    # Plot the image and heatmaps
    img = torch.tensor(img).permute(2, 0, 1) / 255.0
    img_att1 = torch.tensor(img_att1).permute(2, 0, 1) / 255.0
    img_att2 = torch.tensor(img_att2).permute(2, 0, 1) / 255.0
    img_grid = torch.stack([img, img_att1, img_att2])
    torchvision.utils.save_image(img_grid, filepath, padding=0, nrow=3)
