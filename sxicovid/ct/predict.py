import argparse
import json
import os
import torch
from sklearn import metrics
from sklearn.metrics import classification_report

from sxicovid.ct.dataset import load_datasets
from sxicovid.ct.models import CTNet
from sxicovid.utils.plot import save_attention_map

MODELS_PATH = 'ct-models'
MODEL_NAME = 'ct-resnet50-att2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    _, _, test_data = load_datasets(num_classes=3)

    # Instantiate the model and load from folder
    model = CTNet(num_classes=3)
    state_filepath = os.path.join(MODELS_PATH, MODEL_NAME + '.pt')
    model.load_state_dict(torch.load(state_filepath))

    # Get the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Test using device: ' + str(device))

    # Move the model to device
    model.to(device)

    # Make sure the model is set to evaluation mode
    model.eval()

    # Make the prediction
    y_pred = []
    y_true = []
    with torch.no_grad():
        for idx, (example, label) in enumerate(test_data):
            example = example.unsqueeze(0)
            example = example.to(device)
            pred, map1, map2 = model(example, attention=True)
            pred = torch.log_softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1).item()
            y_pred.append(pred)
            y_true.append(label)
            if idx < 1000:
                example = (example + 1) / 2
                save_attention_map(str(idx), example, map1, map2)

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = metrics.confusion_matrix(y_true, y_pred)
    metrics = {'report': report,
               'confusion_matrix': cm.tolist()}

    with open(os.path.join(MODELS_PATH, MODEL_NAME) + '.json', 'w') as file:
        json.dump(metrics, file, indent=4)
