import os
import numpy as np
import cv2
from tqdm import tqdm

def build_pseudo_images(np_folder, out_folder, target_size=224, max_frames=16):
    """Convert 3D landmark sequences to pseudo-images.

    - Input: (T, 68, 3)
    - Resize each frame to (target_size, target_size)
    - Repeat to 3 channels
    - Stack up to max_frames frames (padding/truncation)
    - Save as .npz containing 'x' array of shape (T, H, W, C)
    """
    os.makedirs(out_folder, exist_ok=True)
    for fname in tqdm(sorted(os.listdir(np_folder))):
        if not fname.endswith('.npy'):
            continue
        arr = np.load(os.path.join(np_folder, fname))  # (T, 68, 3) or (T, N)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got {arr.shape}")
        T = arr.shape[0]
        frames = []
        for t in range(min(T, max_frames)):
            frame = arr[t]  # (68,3)
            frame_resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            frame_3ch = np.repeat(frame_resized[:, :, None], 3, axis=2)
            frames.append(frame_3ch)
        if len(frames) < max_frames:
            pad = np.zeros((max_frames - len(frames), target_size, target_size, 3), dtype=np.float32)
            frames = frames + [pad[i] for i in range(len(pad))]
        x = np.stack(frames).astype(np.float32)
        out_path = os.path.join(out_folder, fname.replace('.npy', '.npz'))
        np.savez_compressed(out_path, x=x)
