import pydicom
from pydicom import *
from pathlib import Path
from typing import Optional
from tqdm import auto


def remove_info(dataset,
                data_element,
                va_type=["PN", "LO", "SH", "AE", "DT", "DA"],
                tags=[(0x0010, 0x0040),  # sex
                      (0x0002, 0x0016)  # AE title
                      ],
                update: Optional[dict] = None,
                tags_2_spare=None):
    # Spare sequence name
    if data_element.tag in tags_2_spare:
        return

    # Delete by value group
    if data_element.VR in va_type:
        data_element.value = "Annonymized"

    # Delete by tag
    if data_element.tag in tags:
        data_element.value = ""

    if not update is None:
        keylist = list(update.keys())
        if data_element.tag in list(update.keys()):
            data_element.value = update[data_element.tag]


def annonymize(folder, out_folder, tags=None, update: Optional[dict] = None, tags_2_spare: Optional[dict] = None,
               **kwargs):
    r"""This function annonymize the folder. The target


    ..note::
        If you are using update, the regular paranthesis don't work in dictionary and
        will be converted to integer. You should use the format
        ```
        from pydicom.tag import Tag
        update = {
            Tag((0x0010, 0x0020)): "New name"
        }
        ```
        for this to work.


    Args:
        folder:
        out_folder:
        **kwargs:

    Returns:

    Examples:
        >>> folder, out_folder = "Path to folder", "Path to output folder"
        >>> tags = [
        >>>     (0x0010, 0x0010),  # Patient's Name
        >>>     (0x0010, 0x0020),  # Patient ID
        >>>     (0x0010, 0x0030),  # Patient's Birth Date
        >>>     (0x0010, 0x0040),  # Patient's Sex
        >>>     (0x0010, 0x1040),  # Patient's Address
        >>>     (0x0010, 0x2154),  # Patient's Phone Number
        >>>     (0x0008, 0x0050),  # Accession Number
        >>>     (0x0020, 0x0010),  # Study ID
        >>>     (0x0008, 0x0080),  # Institution Name
        >>>     (0x0008, 0x0081),  # Institution Address
        >>>     (0x0008, 0x0090),  # Referring Physician's Name
        >>>     (0x0008, 0x1048),  # Physician(s) of Record
        >>>     (0x0008, 0x1050),  # Performing Physician's Name
        >>>     (0x0008, 0x1070),  # Operator's Name
        >>>     (0x0010, 0x1090),  # Medical Record Locator
        >>>     (0x0010, 0x21B0),  # Additional Patient History
        >>>     (0x0010, 0x4000),  # Patient Comments
        >>>     (0x0032, 0x1032),  # Requesting Physician
        >>> ]
        >>> update {
        >>>     (0x0010, 0x0020):  "BlahBlah"# Update Name
        >>> }
        >>> annonymize(folder, out_folder, tags, update=update)

    """
    # Default tags to remove for anonymization
    if tags is None:
        tags = [
            (0x0010, 0x0010),  # Patient's Name
            (0x0010, 0x0020),  # Patient ID
            (0x0010, 0x0030),  # Patient's Birth Date
            (0x0010, 0x0040),  # Patient's Sex
            (0x0010, 0x1040),  # Patient's Address
            (0x0010, 0x2154),  # Patient's Phone Number
            (0x0008, 0x0050),  # Accession Number
            (0x0020, 0x0010),  # Study ID
            (0x0008, 0x0080),  # Institution Name
            (0x0008, 0x0081),  # Institution Address
            (0x0008, 0x0090),  # Referring Physician's Name
            (0x0008, 0x1048),  # Physician(s) of Record
            (0x0008, 0x1050),  # Performing Physician's Name
            (0x0008, 0x1070),  # Operator's Name
            (0x0010, 0x1090),  # Medical Record Locator
            (0x0010, 0x21B0),  # Additional Patient History
            (0x0010, 0x4000),  # Patient Comments
            (0x0032, 0x1032),  # Requesting Physician
        ]

    all_dicom_files = list(Path(folder).glob("*.dcm"))
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True)
    out_path = [out_folder.joinpath(r.name) for r in all_dicom_files]

    for fname, oname in auto.tqdm(list(zip(all_dicom_files, out_path))):
        try:
            f = pydicom.dcmread(str(fname))
            f.remove_private_tags()
            f.walk(lambda x1, x2: remove_info(x1, x2, tags=tags, va_type=[], update=update, tags_2_spare=tags_2_spare))
            oname.parent.mkdir(parents=True, exist_ok=True)
            out_name = str(oname)
            f.save_as(out_name)
        except InvalidDicomError:
            print(f"Error when reading: {f}")
    return 0

