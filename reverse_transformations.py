# IMPORTS
import numpy as np
import nibabel as nib
from pathlib import Path
import scipy
import argparse
import os
from utils import map_, tqdm_


def reshape_heart_patient(id_: str, path: Path) -> None:
    # GETTING THE FILE
    GT = nib.load(path / "train" / id_ / "GT.nii.gz")
    image = GT.get_fdata().astype(np.uint8)

    # EXTRACT ONLY THE HEART
    heart_class: int = 2 # assumption, subject to change
    heart_mask: np.ndarray = (image == heart_class).astype(np.uint8) * heart_class

    # THE MATRIX
        # APPARENLY REAL ORDER WAS T4 -> T3 -> R2 -> T1
        # => REVERSE IS T1^-1 -> R2^-1 -> T1 -> T4^-1
    fi = - (27 / 180) * np.pi
    T1: np.ndarray = np.array([[1, 0, 0, 275], [0, 1, 0, 200], [0, 0, 1, 0], [0, 0, 0, 1]])
    R2: np.ndarray = np.array([[np.cos(fi), -np.sin(fi), 0, 0], [np.sin(fi), np.cos(fi), 0, 0], \
        [0, 0, 1, 0], [0, 0, 0, 1]])
    T4: np.ndarray = np.array([[1, 0, 0, 50], [0, 1, 0, 40], [0, 0, 1, 15], [0, 0, 0, 1]])
    matrix: np.ndarray = np.linalg.inv(T1) @ np.linalg.inv(R2) @ T1 @ np.linalg.inv(T4) 

    # TRANSFORMATION -> should it be per slice or can it be total?
    new_image: np.ndarray = scipy.ndimage.affine_transform(heart_mask, matrix, order=0) \
        .astype(np.uint8)

    # PLACE BACK
    new_image += (image - heart_mask)

    # SAVE
    new_image = nib.Nifti1Image(new_image, GT.affine)
    new_image.to_filename(path / "train" / id_ / "GT_fixed.nii.gz")


def main(args: argparse.Namespace):
    # GET PATH
    path: Path = Path(args.path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    assert path.exists()

    # APPLY FUNCTION FOR EACH PATIENT
    for patient in tqdm_([*os.walk(path / "train")][0][1]):
        reshape_heart_patient(patient, path)



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Slicing parameters")
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    main(get_args())