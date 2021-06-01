import os
import json
import torch

from models import SXINet
from plot import save_history
from train import train_classifier
from evaluate import test_classifier
from dataset import load_train_valid_datasets, load_test_dataset

EXPERIMENTS_PATH = 'experiments'


def run_experiment(model_name, equalize=False):
    # Instantiate the model
    model = SXINet(base=model_name.lower())
    print(model)

    # Load the datasets
    train_data, valid_data = load_train_valid_datasets(equalize=equalize)
    test_data = load_test_dataset(equalize=equalize)

    # Set the learning rate according to the model
    if model_name in ['AlexNet', 'VGG16']:
        lr = 1e-4
    else:
        lr = 1e-3

    # Train the classifier
    history = train_classifier(
        model, train_data, valid_data, batch_size=32,
        lr=lr, optimizer='adam', epochs=100, n_workers=4, patience=20
    )

    # Create the experiments directory
    if equalize:
        experiment_path = os.path.join(EXPERIMENTS_PATH, 'with-equalization')
    else:
        experiment_path = os.path.join(EXPERIMENTS_PATH, 'without-equalization')
    try:
        os.mkdir(experiment_path)
    except OSError:
        pass

    # Test the classifier and save the classification report to a JSON file
    report, errors = test_classifier(model, test_data, batch_size=32)
    with open(os.path.join(experiment_path, 'SXINet-' + model_name + '-report.json'), 'w') as file:
        json.dump(report, file, indent=4)

    # Save both the model and training history
    torch.save(model.state_dict(), os.path.join(experiment_path, 'SXINet-' + model_name + '.pt'))
    save_history(history, os.path.join(experiment_path, 'SXINet-' + model_name + '-history.png'))


if __name__ == '__main__':
    if not os.path.isdir(EXPERIMENTS_PATH):
        os.mkdir(EXPERIMENTS_PATH)

    for model_name in ['AlexNet', 'VGG16', 'ResNet50', 'DenseNet121', 'InceptionV3']:
        run_experiment(model_name, equalize=False)
        run_experiment(model_name, equalize=True)
