# IMPORTS
import numpy as np
import nibabel as nib
from pathlib import Path
import scipy
import argparse
import os
from utils import tqdm_


def reshape_heart_patient(path: Path) -> None:
    """
    Function that reverses the transformation applied on the ground truth segmentation
    for a specific patient, only on the heart class and saves the correct images in the 
    same folder (without returning anything).
    Parameters: patient folder path
    """

    # OPEN THE FILE
    GT = nib.load(path / "GT.nii.gz")
    image: np.ndarray = GT.get_fdata().astype(np.uint8)

    # EXTRACT ONLY THE HEART
    heart_class: int = 2 # assumption that turned out to be correct
    heart_mask: np.ndarray = (image == heart_class).astype(np.uint8) * heart_class

    # TRANSFORMATION MATRICES & REVERSE OPERATION
    fi: float = - (27 / 180) * np.pi
    T1: np.ndarray = np.array([[1, 0, 0, 275], [0, 1, 0, 200], [0, 0, 1, 0], [0, 0, 0, 1]])
    R2: np.ndarray = np.array([[np.cos(fi), -np.sin(fi), 0, 0], [np.sin(fi), np.cos(fi), 0, 0], \
        [0, 0, 1, 0], [0, 0, 0, 1]])
    T4: np.ndarray = np.array([[1, 0, 0, 50], [0, 1, 0, 40], [0, 0, 1, 15], [0, 0, 0, 1]])
    matrix: np.ndarray = np.linalg.inv(T4) @ T1 @ np.linalg.inv(R2) @ np.linalg.inv(T1) 

    # TRANSFORMATION
    new_image: np.ndarray = scipy.ndimage.affine_transform(heart_mask, matrix, order=0) \
        .astype(np.uint8)

    # PLACE BACK IN ORIGINAL IMAGE
    new_image += (image - heart_mask) # remove old heart & add new heart simultaneously

    # SAVE NEW IMAGE
    new_image = nib.Nifti1Image(new_image, GT.affine)
    new_image.to_filename(path / "GT_fixed.nii.gz")


def main(args: argparse.Namespace):
    """
    Main function, which applies the function defined above for each patient in the folder.
    """

    # GET PATH
    path: Path = Path(args.path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    assert path.exists()

    # APPLY FUNCTION FOR EACH PATIENT
    for patient in tqdm_([*os.walk(path / "train")][0][1]):
        reshape_heart_patient(path / "train" / patient)


def get_args() -> argparse.Namespace:
    """
    Arguments: dataset path
    """
    parser = argparse.ArgumentParser(description = "Slicing parameters")
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    main(get_args())