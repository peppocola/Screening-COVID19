import os

from sxicovid.ct.models import CTNet
from sxicovid.ct.dataset import load_datasets
from sxicovid.utils.train import train_classifier
from sxicovid.utils.evaluate import test_classifier

EXPERIMENTS_PATH = 'ct-experiments'


if __name__ == '__main__':
    if not os.path.isdir(EXPERIMENTS_PATH):
        os.mkdir(EXPERIMENTS_PATH)

    # Load the datasets
    train_data, valid_data, test_data = load_datasets(n_classes=3)

    # Instantiate the model
    model = CTNet(base='resnet50', n_classes=3)
    print(model)

    batch_size = 64

    train_classifier(
        model, train_data, valid_data, chkpt_path='ct-models/ct-resnet50.pt',
        lr=1e-3, optimizer='adam', batch_size=batch_size, epochs=100, patience=10,
        steps_per_epoch=500, n_workers=2
    )

    report, _ = test_classifier(
        model, test_data, batch_size=batch_size, n_workers=2
    )

    print(report)
