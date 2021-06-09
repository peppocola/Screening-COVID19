import os
import matplotlib.pyplot as plt

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
