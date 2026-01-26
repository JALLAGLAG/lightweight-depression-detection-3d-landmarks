import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_openface(file_path):
    df = pd.read_csv(file_path, sep=",", engine='python')
    return df.values.astype(np.float32)

def extract_openface_all(root_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in tqdm(sorted(os.listdir(root_folder))):
        if fname.endswith("_OpenFace2.1.0_Pose_gaze_AUs.txt"):
            arr = load_openface(os.path.join(root_folder, fname))
            np.save(os.path.join(output_folder, fname.replace('.txt', '.npy')), arr)
