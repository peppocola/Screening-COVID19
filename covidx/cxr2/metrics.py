import os
import time
import torch
import json
import argparse

from covidx.cxr2.models import CXR2Net
from covidx.cxr2.dataset import load_test_dataset, load_train_valid_datasets
from covidx.utils.evaluate import test_classifier
from covidx.utils.plot import plot_cxr2_errors

MODELS_PATH = 'cxr2-models'


def load_model(model_name, equalize=False):
    full_model_name = 'CXR2Net-' + model_name
    if equalize:
        state_filepath = os.path.join(MODELS_PATH, 'with-equalization')
    else:
        state_filepath = os.path.join(MODELS_PATH, 'without-equalization')
    state_filepath = os.path.join(state_filepath, full_model_name, full_model_name + '.pt')
    model = CXR2Net(base=model_name.lower())
    model.load_state_dict(torch.load(state_filepath))
    return model


def load_datasets(equalize=False):
    # Load the datasets, specifically for evaluating metrics
    train_data, valid_data = load_train_valid_datasets(equalize=equalize, augment=False)
    test_data = load_test_dataset(equalize=equalize)
    return {
        'train': train_data,
        'validation': valid_data,
        'test': test_data
    }


def compute_inference_time(model, data, device=None):
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Move the model to device
    model.to(device)

    # Make sure the model is set to evaluation mode
    model.eval()

    # Probably it's better to compute the whole elapsed time
    # instead of computing elapsed times at each iteration and averaging them
    start_time = time.perf_counter()
    with torch.no_grad():
        for inputs, _ in data:
            inputs = inputs.to(device)
            outputs = torch.log_softmax(model(inputs), dim=1)
            torch.argmax(outputs, dim=1)
    stop_time = time.perf_counter()
    return (stop_time - start_time) / len(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str, help='The name of the model to use.'
    )
    parser.add_argument(
        '--equalize', action='store_true', help='Whether to use equalization or not'
    )

    args = parser.parse_args()

    model_name = args.model
    full_model_name = 'CXR2Net-' + model_name
    # Some settings
    batch_size = 4                # Specify a small batch size, used for testing
    time_device = 'cuda'          # Change this to 'cpu' in order to force cpu usage when evaluating inference time
    equalize = args.equalize      # Change this to true to enable histogram equalization

    # Load the model
    model = load_model(model_name, equalize=equalize)

    # Load the datasets
    datasets = load_datasets(equalize=equalize)

    metrics = {
        'n_params': 0,
        'inf_time': {'cpu': 0.0, 'cuda': 0.0},
        'performance': dict()
    }

    # Evaluate the performance metrics
    for split in ['train', 'validation', 'test']:
        report, errors, confusion = test_classifier(
            model, datasets[split], batch_size=batch_size, return_cm=True
        )
        metrics['performance'][split] = {
            'report': report,
            'confusion': confusion.tolist()
        }

    # Plot errors on test
    plot_cxr2_errors(datasets['test'], errors, model_name)

    # Evaluate the number of parameters metric
    metrics['n_params'] = 1e-6 * sum(p.numel() for p in model.parameters())

    # Evaluate the average inference time for a single image
    metrics['inf_time'][time_device] = 1e3 * compute_inference_time(model, datasets['test'], device=time_device)

    # Save the metrics JSON file
    if equalize:
        metrics_filepath = os.path.join(MODELS_PATH, 'with-equalization')
    else:
        metrics_filepath = os.path.join(MODELS_PATH, 'without-equalization')
    metrics_filepath = os.path.join(metrics_filepath, full_model_name, full_model_name + '-metrics.json')
    with open(metrics_filepath, 'w') as file:
        json.dump(metrics, file, indent=4)
