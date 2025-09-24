from pathlib import Path
import numpy as np
import nibabel as nib
import skimage
import os
import argparse
from utils import tqdm_

def stitch_patient(id: str, source_dir: Path, dest_dir: Path):
    # TODO: unite slices in 3D
    # TODO: save in correct location
    slices: list = []
    for file in [*os.walk(source_dir)][0][-1]:
        if id in file:
            slices.append(skimage.io.imread(source_dir / file))
    slices = skimage.io.concatenate_images(slices)

    pass


def main(args: argparse.Namespace):
    """
    Main function, which applies the function defined above for each patient in the folder.
    """

    # GET PATH
    path: Path = Path(args.data_folder)
    assert path.exists()

    # APPLY FUNCTION FOR EACH PATIENT
    for patient in tqdm_([*os.walk(path)][0][1]):
        # TODO: properly apply function with all parameters
        pass


def get_args() -> argparse.Namespace:
    """
    Arguments: dataset path
    """
    # TODO: understand all parameters
    parser = argparse.ArgumentParser(description = "Slicing parameters")
    parser.add_argument('--data_folder', type=str, required=True, help="name of the data folder with sliced data, eg data/prediction/best_epoch/val")
    parser.add_argument('--dest_folder', type=str, required=True, help="name of the destination folder with stitched data, eg val/pred")
    parser.add_argument('--num_classes', type=int, required=True, help="number of classes")
    parser.add_argument('--grp_regex', type=str, required=True, help="pattern for the filename, eg '(Patient_\d\d)_\d\d\d\d'")
    parser.add_argument('--source_scan_pattern', type=str, required=True, help="pattern to the original scans to get original size, eg 'data/train/train/{id_}/GT.nii.gz' (with {id_} to be replaced in stitch.py by the PatientID)")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    main(get_args())