import os
import matplotlib.pyplot as plt

from dataset import CXR2Dataset

IMG_PATH = '../img/'


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


def plot_errors(dataset: CXR2Dataset, errors, model_name):
    label_names = ['negative', 'positive']

    try:
        os.mkdir(IMG_PATH + model_name)
    except OSError:
        pass

    for (idx, label) in errors:
        tensor, _ = dataset.__getitem__(idx)
        img = tensor.numpy().squeeze()

        plt.figure(figsize=(4, 4))
        plt.title(label_names[label])

        path = IMG_PATH + model_name + '/' 'predict_error' + str(idx) + '_' + label_names[label] + '.png'

        plt.imsave(path, img, cmap='gray', format='png')
        plt.close()
