import os
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image as pil


def image_preprocess(img, greyscale, crop, size, remove_top):
    # Convert the image to greyscale
    if greyscale:
        img = img.convert(mode='L')

    # Center crop the image
    if crop:
        width, height = img.size
        if remove_top:
            delta = int(height * 0.08)  # Take the 8% of the top area of the image
            if width > height:
                x, y, d = (width - height + delta) // 2, delta, height - delta
            else:
                x, y, d = delta // 2, delta + (height - width) // 2, width - delta
        else:
            if width > height:
                x, y, d = (width - height) // 2, 0, height
            else:
                x, y, d = 0, (height - width) // 2, width
        assert x >= 0 and y >= 0
        assert x + d <= width and y + d <= height
        box = (x, y, x + d, y + d)
        img = img.crop(box)

    # Resize the image (use bicubic resampling)
    img = img.resize(size, resample=pil.BICUBIC)
    return img


if __name__ == '__main__':
    # Usage example:
    #   python sxicovid/cxr2/preprocessing.py /hdd/Datasets/covidx-cxr2/train.txt --src-path /hdd/Datasets/covidx-cxr2/train \
    #       --greyscale --crop --size 224 224 --remove-top --dest-path datasets/covidx-cxr2/train

    # Instantiate the command line arguments parser
    parser = argparse.ArgumentParser(description='CXR2 Image dataset preprocessor')
    parser.add_argument(
        'labels', type=str, help='The labels text filepath.'
    )
    parser.add_argument(
        '--src-path', type=str, default='.', help='The input dataset path.'
    )
    parser.add_argument(
        '--greyscale', action='store_true', help='Whether to convert images to greyscale.'
    )
    parser.add_argument(
        '--crop', action='store_true', help='Whether to center crop the images.'
    )
    parser.add_argument(
        '--size', nargs=2, type=int, default=(224, 224), help='The size of the output images.'
    )
    parser.add_argument(
        '--remove-top', action='store_true', help='Whether to remove a 8% area from the top of the image.'
    )
    parser.add_argument(
        '--format', choices=['png', 'jpg'], default='png', help='The images output format.'
    )
    parser.add_argument(
        '--dest-path', type=str, default='.', help='The output dataset path.'
    )
    args = parser.parse_args()

    # Create the images directory
    images_path = os.path.join(args.dest_path, 'images')
    if not os.path.isdir(images_path):
        os.makedirs(images_path)

    # Load the labels CSV
    labels_filename, _ = os.path.splitext(os.path.basename(args.labels))
    column_names = ['patient_id', 'filename', 'class', 'data_source']
    df = pd.read_csv(
        args.labels,
        sep=' ',
        names=column_names,
        converters={'class': lambda c: 1 if c == 'positive' else 0}
    )

    # Initialize the preprocessed output CSV
    out_df = pd.DataFrame(columns=column_names)

    for idx, example in tqdm(df.iterrows(), total=len(df)):
        filepath = example['filename']
        filename, ext = os.path.splitext(filepath)

        # Check the extension of the files
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        # Build the input and output image paths
        img_filepath = os.path.join(args.src_path, filepath)
        out_filename = '{}.'.format(idx) + args.format
        out_img_filepath = os.path.join(args.dest_path, 'images', out_filename)

        with pil.open(img_filepath) as img:
            # Preprocess the image
            output_img = image_preprocess(img, args.greyscale, args.crop, args.size, args.remove_top)

            # Save the image
            params = {'quality': 90} if args.format == 'jpg' else dict()
            output_img.save(out_img_filepath, **params)

            # Append a data row to the preprocessed output dataframe
            out_df = out_df.append({
                'patient_id': example['patient_id'],
                'filename': out_filename,
                'class': example['class'],
                'data_source': example['data_source']
            }, ignore_index=True)

    # Save the preprocessed output dataframe
    out_df_filepath = os.path.join(args.dest_path, labels_filename + '.csv')
    out_df.to_csv(out_df_filepath, index=False)
