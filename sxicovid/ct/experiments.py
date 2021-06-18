from sxicovid.ct.models import CTNet
from sxicovid.ct.dataset import load_datasets
from sxicovid.utils.train import train_classifier


if __name__ == '__main__':
    # Load the datasets
    train_data, valid_data, test_data = load_datasets()

    # Instantiate the model
    model = CTNet(num_classes=3)
    print(model)

    # Train the classifier
    train_classifier(
        model, train_data, valid_data, chkpt_path='ct-checkpoints/ct-resnet50-att2.pt',
        lr=5e-4, optimizer='adam', batch_size=64, epochs=25, patience=5,
        steps_per_epoch=1000, n_workers=2
    )
