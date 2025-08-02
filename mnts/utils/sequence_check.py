import re

def unify_mri_sequence_name(m: str, glob_techniques: bool = False, return_glob_dict: bool = False) -> str:
    r"""This function normalize the sequence name from the description tag to a standard system of squence
    identities. Typically, the description tag is the tag (0008|103e) for MRI scans. The description is
    processed with `re` to find what is the sequence identity. Definitions below

    .. warning::
        This function is for MRI only.

    .. note::
        Contrast Mechanism:
            T1W    : T1-weighted image
            T2W    : T2-weighted image
            DWI    : Diffusion-weighted imaging
            IVIM   : Intravoxel incoherent motion
            DCE    : Dynamic contrast-enhanced imaging
            SURVEY : Survey imaging
            eTHRIVE: Enhanced T1 High-Resolution Isotropic Volume Examination (a specialized fat-saturation technique)

        Planes:
            TRA: Axial (Transverse)
            COR: Coronal
            SAG: Sagittal

        Contrast:
            +C/C+: With contrast agent
            FS   : Fat-saturated

        Techniques:
            FLAIR: Fluid-attenuated inversion recovery
            GRE: Gradient echo
            SE: Spin echo
            FSE/TSE: Fast spin echo/Turbo spin echo
            STIR: Short tau inversion recovery
            SWI: Susceptibility-weighted imaging
            EPI: Echo planar imaging
            MPRAGE: Magnetization-prepared rapid gradient echo
            SSFP: Steady-state free precession

    Args:
        m (str):
            Input string from DICOM image description (0008|103e).
        glob_techniques (str, optional):
            If checked, this will also include the technique use in the name.
        return_glob_dict (bool, optional):
            If checked, this will also include the glob dictionary used to process the name.

    Returns:
        str: Unified sequence name

    """
    # This is how you
    weights = {
        'NECK'   : r"(?i)NECK",
        'T1W'    : r"(?i)T1(?!rho)",
        'T2W'    : r"(?i)(T2|longTE)",
        'DWI'    : r"(?i)DWI",
        'IVIM'   : r"(?i)IVIM",
        'DCE'    : r"(?i)DCE",
        'SURVEY' : r"(?i)SURVEY",
        'eTHRIVE': r"(?i)thrive"
    }
    planes = {
        'COR': r"(?i)cor",
        'SAG': r"(?i)sag",
        'TRA': r"(?i)(tra|ax)",
    }
    techniques = {
        'FLAIR': r"(?i)(^|\W|_)flair($|\W|_)",
        'GRE'  : r"(?i)(^|\W|_)gre($|\W|_)",
        'FSE'  : r"(?i)(^|\W|_)fse|tse($|\W|_)",
        'STIR' : r"(?i)(^|\W|_)stir($|\W|_)",
        'SWI'  : r"(?i)(^|\W|_)swi($|\W|_)",
        'EPI'  : r"(?i)(^|\W|_)epi($|\W|_)",
        'SSFP' : r"(?i)(^|\W|_)ssf($|\W|_)",
        'SPIR' : r"(?i)(^|\W|_)spir($|\W|_)",
        'SPAIR': r"(?i)(^|\W|_)spair($|\W|_)"
    }

    contrast = r"(?i)(\+C|C\+|contrast|post)"
    fat_sat = r"(?i)(fs([^e]|$)|fat_saturated|fat\Wsaturated|fat_saturated|fat_sat|fatsat|fat\Wsat|spir|spair|stir|chess)"
    dixon = r"(?i).*dixon.*"    # not used

    """
    Notes: After due consideration, I decide not to add DIXON into directory sorting criteria because
    DIXON is already carried in the file name. The modality is also clinically used as FS technique 
    rather than a unique weighted sequence. 
    """

    w = None    # sequence weight
    p = None    # scan plane
    c = False   # contrast
    fs = False  # fat suppression
    # * contrast weight
    for _w, regpat in weights.items():
        mo = re.search(regpat, m)
        if mo is not None:
            w = _w
            break

    # * planes
    for _p, regpat in planes.items():
        mo = re.search(regpat, m)
        if mo is not None:
            p = _p
            break

    # * techniques
    t = []
    for _t, regpat in techniques.items():
        try:
            mo = re.search(regpat, m)
        except Exception as e:
            print(regpat)
            raise e
        if mo is not None:
            t.append(_t)

    c = re.search(contrast, m) is not None
    fs = re.search(fat_sat, m) is not None
    dx = re.search(dixon, m) is not None

    # reconstruct modality name to unify it
    if w is None:
        w = 'MISC'

    new_name = (f"[{'|'.join(t)}]_" if len(t) and glob_techniques else '') +\
               f"{'CE-' if c else ''}" + \
               f"{w}" + \
               ("-FS" if fs else '') + \
               (f"_{p}" if not p is None else '')

    if return_glob_dict:
        return new_name, {'weight': w,
                          'plane': p,
                          'contrast': c,
                          'fat-suppression': fs,
                          'technique': ','.join(t)}
    else:
        return new_name


def filter_modality(m: str) -> str:
    r"""Determines the modality type from an image description.

    This function parses the image description, typically from the DICOM tag (0008|103e), to classify the image based
    on its sequence weight (e.g., T1W, T2W), plane (e.g., TRA), presence of contrast, and fat suppression. It constructs
    a string that summarizes these attributes.

    Args:
        m (str): The image description string from which to extract modality characteristics.

    Returns:
        str or None: A string indicating the modality type, which might include sequence weight, plane, and markers
        for contrast enhancement (CE) and fat suppression (FS). Returns None if the sequence weight or plane
        cannot be determined. Examples of return values include 'CE-T1W-FS_TRA' or 'T2W_TRA'.

    .. note::
        The regex patterns are case-insensitive and designed to match common abbreviations and terms found in
        medical imaging (e.g., 'tra' for transverse, 'ax' for axial, '+C' or 'C+' for contrast, 'fs' for fat suppression).

        This function only recognizes the first matching pattern for each category (weight, plane, etc.), and it
        stops searching once a match is found.

        If the modality cannot be fully determined (either the weight or the plane is not found), the function
        returns None. This helps in filtering out incomplete or non-standard descriptions.
    """
    weights = {
        'T1W': r"(?i)(.*T1.*)",
        'T2W': r"(?i)(.*T2.*)",
        'DWI': r"(?i)(.*DWI.*)",
        'IVIM': r"(?i)(.*IVIM.*)",
    }
    planes = {
        'TRA': r"(?i)(.*(tra|ax).*)"
    }
    contrast = r"(?i).*(\+C|C\+).*"
    fat_sat = r"(?i).*fs.*"

    w = None    # sequence weight
    p = None    # scan plane
    c = False   # contrast
    fs = False  # fat suppression

    for _w, regpat in weights.items():
        mo = re.match(regpat, m)
        if mo is not None:
            w = _w
            break

    for _p, regpat in planes.items():
        mo = re.match(regpat, m)
        if mo is not None:
            p = _p
            break

    c = re.match(contrast, m) is not None
    fs = re.match(fat_sat, m) is not None

    #
    if w is not None and p is not None:
        filter_type = f"{'CE-' if c else ''}" + \
               f"{w}" + \
               (f"-FS" if fs else "")+ \
               (f"_{p}" if not p is None else "")
    else:
        filter_type = None
    return filter_type