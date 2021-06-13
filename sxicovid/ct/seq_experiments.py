import os

from sxicovid.ct.models import CTSeqNet
from sxicovid.ct.dataset import load_sequence_datasets
from sxicovid.utils.train import train_classifier
from sxicovid.utils.evaluate import test_classifier

EXPERIMENTS_PATH = 'ct-experiments'


if __name__ == '__main__':
    if not os.path.isdir(EXPERIMENTS_PATH):
        os.mkdir(EXPERIMENTS_PATH)

    # Load the datasets
    train_data, valid_data, test_data = load_sequence_datasets(n_classes=3)

    # Instantiate the model
    model = CTSeqNet(input_size=32, pretrained='covidx-ct', n_classes=3)
    print(model)

    batch_size = 8

    train_classifier(
        model, train_data, valid_data, chkpt_path='ct-checkpoints/ct-resnet50-lstm.pt',
        lr=1e-4, optimizer='adam', batch_size=batch_size, epochs=25, patience=5,
        n_workers=2
    )

    report, _ = test_classifier(
        model, test_data, batch_size=batch_size, n_workers=2
    )

    print(report)
