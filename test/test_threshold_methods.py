import SimpleITK as sitk
from mnts.filters.intensity import ThresBinaryClosing
from pathlib import Path



if __name__ == '__main__':
    target_funcs = {
        'HuangThreshold':               lambda x: sitk.HuangThreshold(x, 0, 1),
        'OtsuThreshold':                lambda x: sitk.OtsuThreshold(x, 0, 1),
        'IsoDataThreshold':             lambda x: sitk.IsoDataThreshold(x, 0, 1),
        'MaximumEntropyThreshold':      lambda x: sitk.MaximumEntropyThreshold(x, 0, 1),
        'MomentsThreshold':             lambda x: sitk.MomentsThreshold(x, 0, 1),
        'KittlerIllingworthThreshold':  lambda x: sitk.KittlerIllingworthThreshold(x, 0, 1),
        'LiThreshold':                  lambda x: sitk.LiThreshold(x, 0, 1),
        'ShanbhagThreshold':            lambda x: sitk.ShanbhagThreshold(x, 0, 1),
        'RenyiEntropyThreshold':        lambda x: sitk.RenyiEntropyThreshold(x, 0, 1),
        'YenThreshold':                 lambda x: sitk.YenThreshold(x, 0, 1),
        'TriangleThreshold':            lambda x: sitk.TriangleThreshold(x, 0, 1)
    }

    # test_image = Path(r'../examples/example_data/MRI_01.nii.gz')
    # test_image = Path(r'../examples/example_data/T1w-ce-01.nii.gz')
    # test_image = Path(r'../examples/example_data/T1w-01.nii.gz')
    test_image = Path(r'../examples/example_data/T2w-01.nii.gz')

    for key in target_funcs:
        print(key)
        F = ThresBinaryClosing(closing_kernel_size=10)
        out = Path(r'.').joinpath(key).with_suffix('.nii.gz')
        F.core_func = target_funcs[key]
        out_mask = F.filter(test_image)
        sitk.WriteImage(out_mask, out.resolve().__str__())