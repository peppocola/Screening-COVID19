import argparse
import os
import torch
import shutil

from PIL import Image as pil

from sxicovid.cxr2.metrics import load_model
from sxicovid.cxr2.dataset import load_subset_dataset
from sxicovid.cxr2.preprocessing import image_preprocess

PREPROC_SUBSET_IMAGES_PATH = 'datasets/covidx-cxr2/subset/images'


def preprocess_subset_images(path):
    files = sorted(os.listdir(path))
    try:
        os.makedirs(PREPROC_SUBSET_IMAGES_PATH)
    except OSError:
        shutil.rmtree(PREPROC_SUBSET_IMAGES_PATH)
        os.mkdir(PREPROC_SUBSET_IMAGES_PATH)

    for filename in files:
        filepath = os.path.join(path, filename)
        preproc_filepath = os.path.join(PREPROC_SUBSET_IMAGES_PATH, filename)
        with pil.open(filepath) as img:
            preproc_img = image_preprocess(img, greyscale=True, crop=True, size=(224, 224), remove_top=True)
            preproc_img.save(preproc_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str, help='The name of the model to use.'
    )
    parser.add_argument(
        '--equalize', action='store_true', help='Whether to use equalization or not'
    )
    parser.add_argument(
        '--path', type=str, default='.', help='The path in which the images are stored'
    )

    args = parser.parse_args()

    # Preprocess the dataset
    preprocess_subset_images(path=args.path)

    # Load the model
    equalize = True
    model = load_model(args.model, equalize=args.equalize)

    # Load the dataset
    dataset = load_subset_dataset(equalize=args.equalize)

    # Get the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Move the model to device
    model.to(device)

    # Make sure the model is set to evaluation mode
    model.eval()

    # Setup the data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    # Make the predictions
    y_pred = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = torch.log_softmax(model(inputs), dim=1)
            predictions = torch.argmax(outputs, dim=1)
            y_pred.extend(predictions.cpu().tolist())

    y_pred = ['positive' if x == 1 else 'negative' for x in y_pred]
    result = list(zip(sorted(os.listdir(args.path)), y_pred))
    print(result)
