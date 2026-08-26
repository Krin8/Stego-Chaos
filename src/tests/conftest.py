import os
import sys

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def write_dicom(path, array, slope=1.0, intercept=0.0):
    """Write ``array`` as a minimal but valid single-slice DICOM file."""
    array = np.asarray(array, dtype=np.int16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), Dataset(), file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientID = "test"
    ds.Modality = "CT"
    ds.Rows, ds.Columns = array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.PixelData = array.tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


@pytest.fixture
def dicom_factory(tmp_path):
    """Return a callable creating DICOM files with an HU-like gradient."""

    def factory(name, shape=(16, 16), fill=None, slope=1.0, intercept=-1024.0):
        if fill is None:
            fill = np.linspace(0, 2000, int(np.prod(shape))).reshape(shape)
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return write_dicom(path, fill, slope=slope, intercept=intercept)

    return factory


@pytest.fixture
def read_dicom():
    return pydicom.dcmread
