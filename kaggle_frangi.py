from time import time

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import data

durations = {}
for use_gpu in (False, True):

    if use_gpu:
        from cucim.skimage import color
        from cucim.skimage.filters import meijering, sato, frangi, hessian
        xp = cp
        asnumpy = cp.asnumpy
        device_name = "gpu"
    else:
        from skimage import color
        from skimage.filters import meijering, sato, frangi, hessian
        xp = np
        asnumpy = np.asarray
        device_name = "cpu"

    def identity(image, **kwargs):
        """Return the original image, ignoring any kwargs."""
        return image

    # image = sitk.GetArrayFromImage(sitk.ReadImage("img_for_slide/31000_train.mhd"))
    # image = image[:, 128:192]
    # image = image / 255.0
    # image = xp.asarray(image)
    # image = image.astype(xp.float32)

    retina = data.retina()[200:-200, 200:-200]

    # # transfer image to the GPU
    retina = xp.asarray(retina)
    #Convert RGB image or colormap to grayscale
    image = color.rgb2gray(retina)
    image = image.astype(np.float32)
    # image = cp.tile(image, (4, 4))  # tile to increase size to roughly (4000, 4000)
    # print(f"image.shape = {image.shape}")

    cmap = plt.cm.gray

    kwargs = {'sigmas': range(1, 10), 'mode': 'reflect'}
    fig, axes = plt.subplots(2, 5, figsize=[16, 8])

    tstart = time()
    for i, black_ridges in enumerate([1, 0]):
        for j, func in enumerate([identity, meijering, sato, frangi, hessian]):
            kwargs['black_ridges'] = black_ridges

            result = func(image, **kwargs)

            # transfer back to host for visualization with Matplotlib
            result_cpu = asnumpy(result)
            vmin, vmax = map(float, xp.percentile(result, q=[1, 99.5]))
            axes[i, j].imshow(result_cpu, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            if i == 0:
                axes[i, j].set_title(['Original\nimage', 'Meijering\nneuriteness',
                                      'Sato\ntubeness', 'Frangi\nvesselness',
                                      'Hessian\nvesselness'][j])
            if j == 0:
                axes[i, j].set_ylabel('black_ridges = ' + str(bool(black_ridges)))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    dur = time() - tstart
    print(f"duration = {dur} s")
    durations[device_name] = dur
    plt.tight_layout()
    plt.savefig(f"frangi_{device_name}.png")
    # plt.show()

print(f"GPU Acceleration = {durations['cpu']/durations['gpu']:0.4f}")