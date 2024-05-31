import re

def unify_mri_sequence_name(m: str) -> str:
    r"""This function normalize the sequence name from the description tag to a standard system of squence
    identities. Typically, the description tag is the tag (0008|103e) for MRI scans. The description is
    processed with `re` to find what is the sequence identity. Definitions below

    .. notes::
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

        # TODO: This hasn't been implemented.
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
        m (str): Input string from DICOM image description (0008|103e).

    Returns:
        str: Unified sequence name

    """
    # This is how you
    weights = {
        'NECK'   : r"(?i)NECK",
        'T1W'    : r"(?i)T1",
        'T2W'    : r"(?i)T2",
        'DWI'    : r"(?i)DWI",
        'IVIM'   : r"(?i)IVIM",
        'DCE'    : r"(?i)DCE",
        'SURVEY' : r"(?i)SURVEY",
        'eTHRIVE': r"(?i)e-thrive"
    }
    planes = {
        'COR': r"(?i)cor",
        'SAG': r"(?i)sag",
        'TRA': r"(?i)(tra|ax)",
    }
    contrast = r"(?i)(\+C|C\+|contrast)"
    fat_sat = r"(?i)(fs[^e]|fat_saturated|fat\Wsaturated|fat_saturated|fat_sat|fatsat|fat\Wsat)"
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
    for _w, regpat in weights.items():
        mo = re.search(regpat, m)
        if mo is not None:
            w = _w
            break

    for _p, regpat in planes.items():
        mo = re.search(regpat, m)
        if mo is not None:
            p = _p
            break

    c = re.search(contrast, m) is not None
    fs = re.search(fat_sat, m) is not None
    dx = re.search(dixon, m) is not None

    # reconstruct modality name to unify it
    if w is None:
        w = 'MISC'

    new_name = f"{'CE-' if c else ''}" + \
               f"{w}" + \
               (f"-FS" if fs else "")+ \
               (f"_{p}" if not p is None else "")
    return new_name

def filter_modality(m: str) -> str:
    r"""Check sequence from image description. Usually, this is the from the tag (0008|103e)"""
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