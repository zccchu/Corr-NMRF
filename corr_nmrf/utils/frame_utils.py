import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image
from os.path import splitext

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def readDispPNG(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    return disp, valid


def readDispTiffWHU(filename):
    arr = np.array(Image.open(filename))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint16:
        disp = arr.astype(np.float32) / 256.0
    else:
        disp = arr.astype(np.float32)
    valid = np.isfinite(disp) & (disp > 0.0)
    return disp, valid


def readDispWHU(filename):
    ext = splitext(filename)[-1].lower()
    if ext in (".tif", ".tiff"):
        return readDispTiffWHU(filename)
    return readDispPNG(filename)


def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1].lower()
    if ext in (".png", ".jpeg", ".ppm", ".jpg", ".tif", ".tiff"):
        return Image.open(file_name)
    if ext in (".bin", ".raw"):
        return np.load(file_name)
    return []


class InputPadder:
    """Pad images so their spatial sizes are divisible by the model stride."""

    def __init__(self, dims, mode="proposal", divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == "proposal":
            self._pad = [0, pad_wd, 0, pad_ht]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, *inputs):
        assert all(x.ndim == 4 for x in inputs)
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
