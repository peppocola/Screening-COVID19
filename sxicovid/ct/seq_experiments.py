from sxicovid.ct.models import CTSeqNet
from sxicovid.ct.dataset import load_sequence_datasets
from sxicovid.utils.train import train_classifier


if __name__ == '__main__':
    # Load the datasets
    train_data, valid_data, test_data = load_sequence_datasets(num_classes=3)

    # Instantiate the model
    model = CTSeqNet(input_size=16, num_classes=3, load_embeddings=True)
    print(model)

    batch_size = 8

    train_classifier(
        model, train_data, valid_data, chkpt_path='ct-checkpoints/ct-resnet50-lstm-att2.pt',
        lr=1e-4, optimizer='adam', batch_size=batch_size, epochs=25, patience=5, n_workers=2
    )
