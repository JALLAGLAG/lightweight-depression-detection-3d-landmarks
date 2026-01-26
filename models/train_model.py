import os, glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from .cnn_lstm_model import CNNLSTM
from .losses import FocalLoss

class DepressionDataset(Dataset):
    def __init__(self, file_paths, labels, max_frames=16):
        self.file_paths = file_paths
        self.labels = labels
        self.max_frames = max_frames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        x = data['x']  # (T, H, W, C) for DAIC
        if x.ndim == 4:
            x = np.transpose(x, (0, 3, 1, 2))  # (T, C, H, W)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def load_labels_from_csv(csv_path, prefix_key='participant'):
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Expect columns: participant, label (0/1)
    labels = {row[prefix_key]: int(row['label']) for _, row in df.iterrows()}
    return labels

def get_file_label_pairs(data_folder, labels_dict, ext='.npz'):
    files = []
    labels = []
    for fp in sorted(glob.glob(os.path.join(data_folder, f'*{ext}'))):
        key = os.path.basename(fp).split('.')[0]
        if key in labels_dict:
            files.append(fp)
            labels.append(labels_dict[key])
    return files, labels

def train(config, dataset_name):
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    # Select dataset paths
    if dataset_name == 'daic':
        data_folder = os.path.join(config['daic']['data_root'], config['daic']['pseudo_images_folder'])
        label_csv = os.path.join(config['daic']['data_root'], 'labels_daic.csv')
        ext = '.npz'
    else:
        data_folder = os.path.join(config['edaic']['data_root'], config['edaic']['processed_folder'])
        label_csv = os.path.join(config['edaic']['data_root'], 'labels_edaic.csv')
        ext = '.npz'

    labels_dict = load_labels_from_csv(label_csv)
    files, labels = get_file_label_pairs(data_folder, labels_dict, ext=ext)

    X_train, X_val, y_train, y_val = train_test_split(
        files, labels, test_size=0.2, random_state=config[dataset_name]['split_seed'], stratify=labels
    )

    train_ds = DepressionDataset(X_train, y_train)
    val_ds = DepressionDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])

    model = CNNLSTM(input_channels=3, dropout=config['training']['dropout']).to(device)

    # class weighting
    class_counts = np.bincount(y_train)
    weights = [len(y_train)/c for c in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    focal_loss = FocalLoss(gamma=config['training']['focal_loss_gamma'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    best_f1 = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(config['training']['epochs']):
        model.train()
        train_losses = []
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}'):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = bce_loss(logits, y) + focal_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        y_true, y_scores = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                y_true.extend(y.cpu().numpy().tolist())
                y_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())

        # threshold tuning on validation
        best_thr, best_val_f1 = 0.0, 0.0
        for thr in np.linspace(0.1, 0.9, config['training']['threshold_search_steps']):
            y_pred = [1 if s >= thr else 0 for s in y_scores]
            f1 = f1_score(y_true, y_pred, average='macro')
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_thr = thr

        if best_val_f1 > best_f1:
            best_f1 = best_val_f1
            torch.save({
                'model_state': model.state_dict(),
                'threshold': best_thr,
                'f1_macro': best_f1
            }, f'checkpoints/best_{dataset_name}.pth')

    print(f'Best validation F1-macro: {best_f1:.4f}')
    return best_f1
