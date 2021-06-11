import os
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image as pil


if __name__ == '__main__':
    # Usage example:
    #   python sxicovid/ct/preprocessing.py /hdd/Datasets/covidx-ct/train_COVIDx_CT-2A.txt /hdd/Datasets/covidx-ct/val_COVIDx_CT-2A.txt \
    #       /hdd/Datasets/covidx-ct/test_COVIDx_CT-2A.txt --src-path /hdd/Datasets/covidx-ct/2A_images --size 224 224 --dest-path datasets/covidx-ct

    # Instantiate the command line arguments parser
    parser = argparse.ArgumentParser(description='CT Image dataset preprocessor')
    parser.add_argument(
        'train_labels', type=str, help='The train labels text filepath.'
    )
    parser.add_argument(
        'valid_labels', type=str, help='The validation labels text filepath.'
    )
    parser.add_argument(
        'test_labels', type=str, help='The test labels text filepath.'
    )
    parser.add_argument(
        '--src-path', type=str, default='.', help='The input dataset path.'
    )
    parser.add_argument(
        '--size', nargs=2, type=int, default=(224, 224), help='The size of the output images.'
    )
    parser.add_argument(
        '--dest-path', type=str, default='.', help='The output dataset path.'
    )
    args = parser.parse_args()

    # Set the random seed
    np.random.seed(42)

    # Create the images directories
    train_images_path = os.path.join(args.dest_path, 'train')
    if not os.path.isdir(train_images_path):
        os.makedirs(train_images_path)
    valid_images_path = os.path.join(args.dest_path, 'valid')
    if not os.path.isdir(valid_images_path):
        os.makedirs(valid_images_path)
    test_images_path = os.path.join(args.dest_path, 'test')
    if not os.path.isdir(test_images_path):
        os.makedirs(test_images_path)

    # Load the labels CSVs
    labels_column_names = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    train_labels_filename, _ = os.path.splitext(os.path.basename(args.train_labels))
    valid_labels_filename, _ = os.path.splitext(os.path.basename(args.valid_labels))
    test_labels_filename, _ = os.path.splitext(os.path.basename(args.test_labels))
    dataset_labels = {
        'train': pd.read_csv(args.train_labels, sep=' ', names=labels_column_names),
        'valid': pd.read_csv(args.valid_labels, sep=' ', names=labels_column_names),
        'test': pd.read_csv(args.test_labels, sep=' ', names=labels_column_names)
    }

    # Initialize the preprocessed output CSV
    out_labels_column_names = ['filename', 'class']
    out_dataset_labels = {
        'train': pd.DataFrame(columns=out_labels_column_names),
        'valid': pd.DataFrame(columns=out_labels_column_names),
        'test': pd.DataFrame(columns=out_labels_column_names)
    }

    # Process rows
    for slice in ['train', 'valid', 'test']:
        df = dataset_labels[slice]
        for idx, sample in tqdm(df.iterrows(), total=len(df)):
            filename = sample['filename']
            target = sample['class']
            box = (sample['xmin'], sample['ymin'], sample['xmax'], sample['ymax'])

            # Build the image filepath
            out_filename = '{}.png'.format(idx)
            img_filepath = os.path.join(args.src_path, filename)
            with pil.open(img_filepath) as img:
                # Preprocess the image
                img = img.convert(mode='L').crop(box).resize(args.size, resample=pil.BICUBIC)

            # Save the PNG image
            out_filepath = os.path.join(args.dest_path, slice, out_filename)
            img.save(out_filepath)

            # Append a data row to the preprocessed output CSV
            out_dataset_labels[slice] = out_dataset_labels[slice].append({
                'filename': out_filename,
                'class': target
            }, ignore_index=True)

        out_df_filepath = os.path.join(args.dest_path, '{}.csv'.format(slice))
        out_dataset_labels[slice].to_csv(out_df_filepath, index=False)
