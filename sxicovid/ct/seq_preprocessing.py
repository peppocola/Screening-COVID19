import os
import re
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image as pil


if __name__ == '__main__':
    # Usage example:
    #   python sxicovid/ct/preprocessing.py /hdd/Datasets/covidx-ct/train_COVIDx_CT-2A.txt /hdd/Datasets/covidx-ct/val_COVIDx_CT-2A.txt \
    #       /hdd/Datasets/covidx-ct/test_COVIDx_CT-2A.txt --src-path /hdd/Datasets/covidx-ct/2A_images --size 224 224 --ct-length 32 --dest-path datasets/covidx-seqct

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
        '--ct-length', type=int, default=32, help='The fixed length of a CT scan.'
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
    for dataset in ['train', 'valid', 'test']:
        ct_infos = []
        prev_ct_id = None
        df = dataset_labels[dataset]
        tk = tqdm(total=len(df))
        idx = 0

        while idx < len(df):
            sample = df.iloc[idx]
            filename = sample['filename']
            target = sample['class']
            box = (sample['xmin'], sample['ymin'], sample['xmax'], sample['ymax'])

            result = re.search('^(.+)[_-]([0-9]+).png$', filename)
            if result is None:
                result = re.search('^(.+)[_-]IM([0-9]+).png$', filename)
            assert result is not None, 'Regex mismatch - {}'.format(filename)
            ct_id = result.group(1)

            if prev_ct_id is None:
                prev_ct_id = ct_id
                ct_infos.clear()

            if prev_ct_id == ct_id:
                filepath = os.path.join(args.src_path, filename)
                ct_infos.append((filepath, box, target))
                idx += 1
                tk.update()
                tk.refresh()
                continue

            ct_id = prev_ct_id
            prev_ct_id = None
            num_ct_images = len(ct_infos)
            if num_ct_images < args.ct_length:
                continue

            ct_slices = []
            for img_filepath, img_box, img_target in ct_infos:
                with pil.open(img_filepath) as img:
                    img = img.convert(mode='L').crop(img_box).resize(args.size, resample=pil.BICUBIC)
                    ct_slices.append((img, img_target))

            out_filename = '{}.tiff'.format(ct_id)
            out_filepath = os.path.join(args.dest_path, dataset, out_filename)
            images, targets = zip(*ct_slices)
            assert len(set(targets)) == 1, 'Targets mismatch - {} : {}'.format(ct_id, targets)
            ct_class = targets[0]

            step_size = num_ct_images // args.ct_length
            sample_indices = np.arange(0, num_ct_images, step=step_size)
            mask = np.random.choice(np.arange(len(sample_indices)), size=args.ct_length, replace=False)
            sample_indices = sample_indices[mask]
            sample_indices = list(sorted(sample_indices))
            filtered_images = []
            for i in sample_indices:
                filtered_images.append(images[i])
            filtered_images[0].save(out_filepath, append_images=filtered_images[1:], save_all=True)

            out_dataset_labels[dataset] = out_dataset_labels[dataset].append({
                'filename': out_filename,
                'class': ct_class
            }, ignore_index=True)

        tk.close()
        out_df_filepath = os.path.join(args.dest_path, '{}.csv'.format(dataset))
        out_dataset_labels[dataset].to_csv(out_df_filepath, index=False)
