import os
import numpy as np
import pandas as pd

def load_clnf(file_path):
    """Load CLNF 3D landmarks file into numpy array.

    The DAIC-WOZ file format is tab-separated. Each row corresponds to one frame.
    The first columns may contain metadata; adjust parsing if needed.
    """
    df = pd.read_csv(file_path, sep="\t", header=None, comment='#', engine='python')
    return df.values.astype(np.float32)

def extract_all_landmarks(root_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in sorted(os.listdir(root_folder)):
        if fname.endswith("_CLNF_features3D.txt"):
            arr = load_clnf(os.path.join(root_folder, fname))
            np.save(os.path.join(output_folder, fname.replace('.txt', '.npy')), arr)
