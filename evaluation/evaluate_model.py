import os, glob
import torch
import numpy as np
from sklearn.metrics import f1_score
from models.cnn_lstm_model import CNNLSTM

def evaluate(config, dataset_name, checkpoint_path):
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    if dataset_name == 'daic':
        data_folder = os.path.join(config['daic']['data_root'], config['daic']['pseudo_images_folder'])
        label_csv = os.path.join(config['daic']['data_root'], 'labels_daic.csv')
        ext = '.npz'
    else:
        data_folder = os.path.join(config['edaic']['data_root'], config['edaic']['processed_folder'])
        label_csv = os.path.join(config['edaic']['data_root'], 'labels_edaic.csv')
        ext = '.npz'

    import pandas as pd
    labels_df = pd.read_csv(label_csv)
    labels_dict = {row['participant']: int(row['label']) for _, row in labels_df.iterrows()}

    files = []
    y_true = []
    for fp in sorted(glob.glob(os.path.join(data_folder, f'*{ext}'))):
        key = os.path.basename(fp).split('.')[0]
        if key in labels_dict:
            files.append(fp)
            y_true.append(labels_dict[key])

    model = CNNLSTM(input_channels=3, dropout=config['training']['dropout']).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    threshold = ckpt['threshold']

    y_scores = []
    model.eval()
    with torch.no_grad():
        for fp in files:
            data = np.load(fp)
            x = data['x']
            x = np.transpose(x, (0, 3, 1, 2))
            x = torch.tensor(x[None], dtype=torch.float32).to(device)
            logits = model(x)
            y_scores.append(torch.sigmoid(logits).item())

    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Evaluation F1-macro: {f1:.4f}')
    return f1
