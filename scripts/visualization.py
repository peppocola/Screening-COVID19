import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from PIL import Image as pil
from PIL import ImageOps as pilops
from preprocessing import image_preprocess

IMAGES_PATH = '/hdd/Datasets/covidx-cxr2/train'
LABELS_FILEPATH = '/hdd/Datasets/covidx-cxr2/train.txt'


def save_visualizations(filepath, identifier, output_dir='visualization'):
    # Check the presence of the output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Obtain the filenames
    original_filename = os.path.join(output_dir, identifier + '.png')
    cropped_filename = os.path.join(output_dir, identifier + '-cropped.png')
    histogram_filename = os.path.join(output_dir, identifier + '-histogram.pdf')
    equalized_filename = os.path.join(output_dir, identifier + '-equalized.png')

    with pil.open(filepath) as img:
        # Save the original image
        img.save(original_filename)

        # Preprocess the image
        img = image_preprocess(img, greyscale=True, crop=True, size=(224, 224), remove_top=True)

        # Save the cropped image
        img.save(cropped_filename)

        # Compute the cumulative histogram
        histogram = img.histogram()
        cumulative_histogram = np.cumsum(histogram) / np.sum(histogram)

        # Equalize the image
        equalized_img = pilops.equalize(img)

        # Save the equalized image
        equalized_img.save(equalized_filename)

        # Save the histogram figure
        xticks = np.arange(5) * 64
        xticks[-1] = 255
        yticks = np.arange(6) / 5.0
        plt.figure(figsize=(3, 3))
        plt.xlim([0, 255])
        plt.ylim([0.0, 1.0])
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.plot(cumulative_histogram, linewidth=2)
        plt.tight_layout()
        plt.savefig(histogram_filename)
        plt.clf()


def save_dimensions_histogram(hist, output_dir='visualization'):
    # Check the presence of the output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Obtain the filenames
    filename = os.path.join(output_dir, 'dataset-dimensions.pdf')

    # Plot the bar diagram of the image dimensions frequencies
    dims, freqs = zip(*hist)
    dims = list(map(lambda x: '{}x{}'.format(x[0], x[1]), dims))
    plt.figure(figsize=(6, 4))
    plt.bar(dims, freqs, log=True, width=0.6)
    plt.xticks(rotation='vertical')
    plt.ylabel('number of examples')
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


if __name__ == '__main__':
    # Set the random seed
    np.random.seed(42)

    # Load the labels CSV
    df = pd.read_csv(
        LABELS_FILEPATH,
        sep=' ',
        names=['patient_id', 'filename', 'class', 'data_source'],
        converters={'class': lambda c: 1 if c == 'positive' else 0}
    )

    negative_df = df[df['class'] == 0]
    positive_df = df[df['class'] == 1]

    # Sample some negative and positive examples randomly
    n_samples = 3
    negative_idx = np.random.choice(len(negative_df), size=n_samples, replace=False)
    positive_idx = np.random.choice(len(positive_df), size=n_samples, replace=False)
    negative_df = negative_df.iloc[negative_idx]
    positive_df = positive_df.iloc[positive_idx]

    # Save the visualizations of those examples
    for idx, filename in enumerate(negative_df['filename']):
        filepath = os.path.join(IMAGES_PATH, filename)
        save_visualizations(filepath, 'cxr2-id{}-negative'.format(idx))
    for idx, filename in enumerate(positive_df['filename']):
        filepath = os.path.join(IMAGES_PATH, filename)
        save_visualizations(filepath, 'cxr2-id{}-positive'.format(idx))

    # Compute the image dimensions histogram
    histogram = defaultdict(int)
    for filename in df['filename']:
        filepath = os.path.join(IMAGES_PATH, filename)
        with pil.open(filepath) as img:
            histogram[img.size] += 1

    # Save the image dimensions histogram
    histogram = list(sorted(
        filter(lambda x: x[1] >= 10, histogram.items()),
        key=lambda x: x[1], reverse=True
    ))
    save_dimensions_histogram(histogram)
