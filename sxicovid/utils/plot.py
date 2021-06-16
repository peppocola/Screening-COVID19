import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

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


def save_attention_map(filename, img, att1, att2):
    path = 'ct-attentions/' + filename + '.png'
    image_size = (224, 224)

    # Remove batch dimension
    img = torch.squeeze(img)
    att1 = torch.squeeze(att1)
    att2 = torch.squeeze(att2)

    # Convert to numpy array
    img = img.cpu().data.numpy()
    att1 = att1.cpu().data.numpy()
    att2 = att2.cpu().data.numpy()

    # Normalize attention map
    att1 = att1 * (1 / att1.max()) * 255
    att2 = att2 * (1 / att2.max()) * 255

    # Upsample the image
    att1 = cv2.resize(att1, image_size, interpolation=cv2.INTER_CUBIC)
    att2 = cv2.resize(att2, image_size, interpolation=cv2.INTER_CUBIC)

    # Thresholding
    _, att1 = cv2.threshold(att1, 32, 255, type=cv2.THRESH_TOZERO)
    _, att2 = cv2.threshold(att2, 128, 255, type=cv2.THRESH_TOZERO)

    att1 = att1.astype(np.uint8)
    att2 = att2.astype(np.uint8)

    # Apply colormap
    att1 = cv2.applyColorMap(att1, cv2.COLORMAP_JET)
    att2 = cv2.applyColorMap(att2, cv2.COLORMAP_JET)

    img = np.expand_dims(img, 2) * 255

    # Combine heatmap and image
    heatmap = 0.6 * img + 0.2 * att1 + 0.2 * att2
    heatmap = heatmap.astype(np.uint8)

    # Save image
    cv2.imwrite(path, heatmap)
