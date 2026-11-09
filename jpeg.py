import os
import numpy as np
import nibabel as nib
from PIL import Image
from fastai.vision.all import *
from fastcore.foundation import patch
import torch
from types import SimpleNamespace

def read_nii(filepath):
    """
    Reads a NIfTI file and returns a rotated numpy array.
    """
    print(f"Loading NIfTI file: {filepath}")
    nii = nib.load(filepath)
    data = nii.get_fdata()
    data = np.rot90(data)  # Adjust orientation if needed
    print("Loaded volume shape:", data.shape)
    return data

# Define window settings
dicom_windows = SimpleNamespace(
    liver = (150, 30),   # (window_width, window_level)
    custom = (200, 60)
)

# Custom Tensor class for CT scans (inherits from fastai's TensorImageBW)
class TensorCTScan(TensorImageBW):
    _show_args = {'cmap': 'bone'}

# ----- fastai patch: apply windowing to a tensor -----
@patch
def windowed(self:Tensor, w, l):
    """
    Applies windowing to a tensor:
    - Clamps the values between (l - w/2) and (l + w/2)
    - Scales the result between 0 and 1.
    """
    px = self.clone()
    px_min = l - w // 2
    px_max = l + w // 2
    px[px < px_min] = px_min
    px[px > px_max] = px_max
    return (px - px_min) / (px_max - px_min)

# ----- fastai patch: convert a single-channel tensor to multi-channel using windows -----
@patch
def to_nchan(x:Tensor, wins, bins=None):
    """
    Converts a single-channel tensor to a multi-channel tensor using
    the provided window settings.
    """
    res = [x.windowed(*win) for win in wins]
    # Append an extra normalized channel if bins is not set to 0
    if not (isinstance(bins, int) and bins == 0):
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        res.append(x_norm.clamp(0,1))
    return TensorCTScan(torch.stack(res, dim=0))

# ----- fastai patch: save a tensor as a JPEG image -----
@patch
def save_jpg(x:Tensor, path, wins, bins=None, quality=90):
    """
    Saves a tensor as a JPEG image after converting it to multi-channel format.
    
    Parameters:
    - path: File path (without extension) where the image will be saved.
    - wins: List of window settings (tuples) to apply.
    - bins: Parameter for extra channel (default adds a normalized channel).
    - quality: JPEG quality.
    """
    fn = Path(path).with_suffix('.jpg')
    x_nchan = x.to_nchan(wins, bins)
    # Scale to 0-255 and convert to uint8
    x_img = (x_nchan * 255).byte()
    # If the image has 3 channels, set mode to 'RGB'; otherwise use 'L' (grayscale)
    mode = 'RGB' if x_img.shape[0] == 3 else 'L'
    im = Image.fromarray(x_img.permute(1, 2, 0).numpy(), mode=mode)
    im.save(fn, quality=quality)
    print(f"Saved image: {fn}")

def save_slices_as_jpg_nii(nii_path, output_dir, wins=[dicom_windows.liver, dicom_windows.custom], step=2):
    """
    Reads a NIfTI file, applies fastai windowing, and saves every 'step'-th slice as JPEG.
    
    Parameters:
    - nii_path: Path to the .nii file.
    - output_dir: Directory where JPEG images will be saved.
    - wins: List of window settings (tuples) to apply.
    - step: Save every 'step'-th slice.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using existing directory: {output_dir}")

    if not os.path.isfile(nii_path):
        print(f"Error: NIfTI file does not exist: {nii_path}")
        return

    volume = read_nii(nii_path)
    num_slices = volume.shape[-1]
    print("Total slices in volume:", num_slices)

    for i in range(0, num_slices, step):
        slice_array = volume[..., i].astype(np.float32)
        slice_tensor = tensor(slice_array)
        filename = os.path.join(output_dir, f"slice_{i:03d}")
        slice_tensor.save_jpg(filename, wins)
    print("Finished saving slices.")

if __name__ == "__main__":
    nii_path = "liver tumor dataset 2/volume_pt4/volume-33.nii"  # <-- Update this path with your .nii file!
    output_dir = "output_slices"        # Directory to save JPEG images
    save_slices_as_jpg_nii(nii_path, output_dir, wins=[dicom_windows.liver, dicom_windows.custom], step=2)
