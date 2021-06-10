import os
import re
import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image as pil


def find_patient_cts(regex):
    train_cts = train_df[train_df['filename'].str.contains(regex)]
    if not train_cts.empty:
        return train_cts, 'train'
    else:
        valid_cts = valid_df[valid_df['filename'].str.contains(regex)]
        if not valid_cts.empty:
            return valid_cts, 'valid'
        else:
            test_cts = test_df[test_df['filename'].str.contains(regex)]
            if not test_cts.empty:
                return test_cts, 'test'
            else:
                return None, None


if __name__ == '__main__':
    # Usage example:
    #   python sxicovid/ct/preprocessing.py /hdd/Datasets/covidx-ct/metadata.csv /hdd/Datasets/covidx-ct/train_COVIDx_CT-2A.txt /hdd/Datasets/covidx-ct/val_COVIDx_CT-2A.txt \
    #       /hdd/Datasets/covidx-ct/test_COVIDx_CT-2A.txt --src-path /hdd/Datasets/covidx-ct/2A_images --size 224 224 --ct-length 32 --dest-path datasets/covidx-ct

    # Instantiate the command line arguments parser
    parser = argparse.ArgumentParser(description='CT Image dataset preprocessor')
    parser.add_argument(
        'metadata', type=str, help='The metadata csv filepath.'
    )
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
        '--ct-length', type=int, default=16, help='The fixed length of a CT scan.'
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
    metadata_df = pd.read_csv(args.metadata, sep=',')
    train_df = pd.read_csv(args.train_labels, sep=' ', names=labels_column_names)
    valid_df = pd.read_csv(args.valid_labels, sep=' ', names=labels_column_names)
    test_df = pd.read_csv(args.test_labels, sep=' ', names=labels_column_names)

    # Initialize the preprocessed output CSV
    out_labels_column_names = ['filename', 'class']
    dataset_labels = {
        'train': pd.DataFrame(columns=out_labels_column_names),
        'valid': pd.DataFrame(columns=out_labels_column_names),
        'test': pd.DataFrame(columns=out_labels_column_names)
    }

    # Process metadata rows
    for idx, example in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        patient_id = example['patient id']

        # Obtain the CTs for a certain patient
        regex = '^{}[_-]'.format(patient_id)
        cts_df, dataset = find_patient_cts(regex)
        if cts_df is None:
            regex = '^.+[_-]{}[_-]'.format(patient_id)
            cts_df, dataset = find_patient_cts(regex)
            if cts_df is None:
                print('Patient {} not found - skipping'.format(patient_id))
                continue

        # Obtain the IDs of all the CTs of a certain patient
        unique_cts = set()
        for _, ct in cts_df.iterrows():
            result = re.search('({})(.+)([_-][0-9]+.png)$'.format(regex), ct['filename'])
            if result is None:
                continue
            unique_cts.add(result.group(2))
        if len(unique_cts) == 0:
            patient_cts = [cts_df]
        else:
            patient_cts = [
                cts_df[cts_df['filename'].str.contains('{}{}'.format(regex, i))]
                for i in unique_cts
            ]

        # Process every CT of a single patient
        for ct_idx, patient_ct in enumerate(patient_cts):
            if len(patient_ct) < args.ct_length:
                continue

            # Sample a fixed subset of CT slices (uniformly)
            step_size = len(patient_ct) // args.ct_length
            sample_indices = np.arange(0, len(patient_ct), step=step_size)
            mask = np.random.choice(np.arange(len(sample_indices)), size=args.ct_length, replace=False)
            sample_indices = sample_indices[mask]
            sample_indices = list(sorted(sample_indices))
            patient_ct.reset_index(inplace=True)

            # Process every slice
            slices = []
            ct_class = patient_ct.iloc[0]['class']
            out_filename = '{}_{}.tiff'.format(patient_id, ct_idx)
            out_filepath = os.path.join(args.dest_path, dataset, out_filename)
            for slice_id, sample_idx in enumerate(sample_indices):
                sample = patient_ct.iloc[sample_idx]
                box = (sample['xmin'], sample['ymin'], sample['xmax'], sample['ymax'])
                filename = sample['filename']

                # Build the image filepath
                img_filepath = os.path.join(args.src_path, filename)
                with pil.open(img_filepath) as img:
                    # Preprocess the image
                    img = img.convert(mode='L').crop(box).resize(args.size, resample=pil.BICUBIC)
                    slices.append(img)

            # Save the multi-channels TIFF image
            slices[0].save(out_filepath, append_images=slices[1:], save_all=True)

            # Append a data row to the preprocessed output CSV
            dataset_labels[dataset] = dataset_labels[dataset].append({
                'filename': out_filename,
                'class': ct_class
            }, ignore_index=True)

    # Save the preprocessed output CSVs
    for dataset in dataset_labels:
        out_df_filepath = os.path.join(args.dest_path, '{}.csv'.format(dataset))
        dataset_labels[dataset].to_csv(out_df_filepath, index=False)
