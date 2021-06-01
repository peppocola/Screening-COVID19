import torch
from sklearn import metrics


def test_classifier(
        model,
        data_test,
        batch_size=128,
        n_workers=4,
        device=None,
        return_cm=False
):
    # Get the device to use
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Move the model to device
    model.to(device)

    # Setup the data loader
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    # Make sure the model is set to evaluation mode
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = torch.log_softmax(model(inputs), dim=1)
            predictions = torch.argmax(outputs, dim=1)
            y_pred.extend(predictions.cpu().tolist())
            y_true.extend(targets.cpu().tolist())

    errors = [(idx, label) for idx, label in enumerate(y_true) if label != y_pred[idx]]

    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    if return_cm:
        cm = metrics.confusion_matrix(y_true, y_pred)
        return report, errors, cm
    return report, errors
