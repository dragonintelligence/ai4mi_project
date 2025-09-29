from pathlib import Path
import numpy as np
import nibabel as nib
import skimage
import os
import argparse
from utils import tqdm_

def stitch_patient(id_: str, source: Path, dest: Path, gt: Path):
    """
    Function that converts the set of 2D slices of a patient into a 3D Nifti file and saves it.
    Parameters: Patient id, source folder, destination folder, 
    """
    # FOR REFERENCE
    original = nib.load(gt)
    size: tuple = original.get_fdata().shape[:2]

    # EXTRACT 2D SLICES
    slices: list = []
    for filename in [*os.walk(source)][0][-1]:
        if id_ in filename:
            slices.append(skimage.io.imread(source / filename))
    slices = skimage.io.concatenate_images(slices)
    
    # TRANSFORM
    slices = np.array([skimage.transform.resize(image, size, preserve_range=True, anti_aliasing=False) \
        for image in slices])
    print(slices.shape)
    slices = (np.transpose(slices, (1, 2, 0)) // 63).astype(np.uint8)
    
    # SAVE INTO 3D FILE
    image = nib.Nifti1Image(slices, original.affine, original.header)
    nib.save(image, dest / f"{id_}.nii.gz")


def main(args: argparse.Namespace):
    """
    Main function, which applies the function defined above for each patient in the folder.
    """

    # GET PATH
    path: Path = Path(args.data_folder)
    assert path.exists()

    # APPLY FUNCTION FOR EACH PATIENT
    patients: list = set([id[:10] for id in [*os.walk(path)][0][-1]])
    for patient in tqdm_(patients):
        stitch_patient(patient, path, Path(args.dest_folder), Path(args.source_scan_pattern.replace\
        ("{id}", patient)))

def get_args() -> argparse.Namespace:
    """
    Arguments: dataset path
    """
    # TODO: understand all parameters
    parser = argparse.ArgumentParser(description = "Slicing parameters")
    parser.add_argument('--data_folder', type=str, required=True, help="name of the data folder with sliced data, eg data/prediction/best_epoch/val")
    parser.add_argument('--dest_folder', type=str, required=True, help="name of the destination folder with stitched data, eg val/pred")
    #parser.add_argument('--num_classes', type=int, required=True, help="number of classes (e.g.: 5)")
    # parser.add_argument('--grp_regex', type=str, required=True, help="pattern for the filename, eg '(Patient_\d\d)_\d\d\d\d'")
    parser.add_argument('--source_scan_pattern', type=str, required=True, help="pattern to the original scans to get original size, eg 'data/train/train/{id_}/GT.nii.gz' (with {id_} to be replaced in stitch.py by the PatientID)")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    main(get_args())