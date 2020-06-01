import argparse
import os

import numpy as np
import cv2
import torch
from pytorch_toolbelt.utils import fs, to_numpy
from tqdm import tqdm

from alaska2 import idct8
from alaska2.dataset import idct8v2
from alaska2.models.dct import SpaceToDepth

sd2 = SpaceToDepth(block_size=8)

import torch
from torch import Tensor
from typing import Iterable


class RunningStatistics:
    """Records mean and variance of the final `n_dims` dimension over other dimensions across items. So collecting across `(l,m,n,o)` sized
       items with `n_dims=1` will collect `(l,m,n)` sized statistics while with `n_dims=2` the collected statistics will be of size `(l,m)`.
       Uses the algorithm from Chan, Golub, and LeVeque in "Algorithms for computing the sample variance: analysis and recommendations":
       `variance = variance1 + variance2 + n/(m*(m+n)) * pow(((m/n)*t1 - t2), 2)`
       This combines the variance for 2 blocks: block 1 having `n` elements with `variance1` and a sum of `t1` and block 2 having `m` elements
       with `variance2` and a sum of `t2`. The algorithm is proven to be numerically stable but there is a reasonable loss of accuracy (~0.1% error).
       Note that collecting minimum and maximum values is reasonably innefficient, adding about 80% to the running time, and hence is disabled by default.
    """

    def __init__(self, n_dims: int = 2, record_range=False):
        self._n_dims, self._range = n_dims, record_range
        self.n, self.sum, self.min, self.max = 0, None, None, None

    def update(self, data: Tensor):
        data = data.view(*list(data.shape[: -self._n_dims]) + [-1])
        with torch.no_grad():
            new_n, new_var, new_sum = data.shape[-1], data.var(-1), data.sum(-1)
            if self.n == 0:
                self.n = new_n
                self._shape = data.shape[:-1]
                self.sum = new_sum
                self._nvar = new_var.mul_(new_n)
                if self._range:
                    self.min = data.min(-1)[0]
                    self.max = data.max(-1)[0]
            else:
                assert (
                    data.shape[:-1] == self._shape
                ), f"Mismatched shapes, expected {self._shape} but got {data.shape[:-1]}."
                ratio = self.n / new_n
                t = (self.sum / ratio).sub_(new_sum).pow_(2)
                self._nvar.add_(new_n, new_var).add_(ratio / (self.n + new_n), t)
                self.sum.add_(new_sum)
                self.n += new_n
                if self._range:
                    self.min = torch.min(self.min, data.min(-1)[0])
                    self.max = torch.max(self.max, data.max(-1)[0])

    @property
    def mean(self):
        return self.sum / self.n if self.n > 0 else None

    @property
    def var(self):
        return self._nvar / self.n if self.n > 0 else None

    @property
    def std(self):
        return self.var.sqrt() if self.n > 0 else None

    def __repr__(self):
        def _fmt_t(t: Tensor):
            if t.numel() > 5:
                return f"tensor of ({','.join(map(str, t.shape))})"

            def __fmt_t(t: Tensor):
                return "[" + ",".join([f"{v:.3g}" if v.ndim == 0 else __fmt_t(v) for v in t]) + "]"

            return __fmt_t(t)

        rng_str = f", min={_fmt_t(self.min)}, max={_fmt_t(self.max)}" if self._range else ""
        return f"RunningStatistics(n={self.n}, mean={_fmt_t(self.mean)}, std={_fmt_t(self.std)}{rng_str})"


def compute_mean_std(dataset):
    """
    https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
    """

    # global_mean = np.zeros((3 * 64), dtype=np.float64)
    # global_var = np.zeros((3 * 64), dtype=np.float64)

    n_items = 0
    s = RunningStatistics()

    for image_fname in dataset:
        dct_file = np.load(fs.change_extension(image_fname, ".npz"))
        y = torch.from_numpy(dct_file["dct_y"])
        cb = torch.from_numpy(dct_file["dct_cb"])
        cr = torch.from_numpy(dct_file["dct_cr"])

        dct = torch.stack([y, cb, cr], dim=0).unsqueeze(0).float()
        dct = sd2(dct)[0]
        s.update(dct)
        # dct = to_numpy()

        # global_mean += dct.mean(axis=(1, 2))
        # global_var += dct.std(axis=(1, 2)) ** 2
        # n_items += 1

    return s.mean, s.std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, default=os.environ.get("KAGGLE_2020_ALASKA2"))

    args = parser.parse_args()
    data_dir = args.data_dir

    cover = os.path.join(data_dir, "Cover")
    JMiPOD = os.path.join(data_dir, "JMiPOD")
    JUNIWARD = os.path.join(data_dir, "JUNIWARD")
    UERD = os.path.join(data_dir, "UERD")

    dataset = (
        fs.find_images_in_dir(cover)
        + fs.find_images_in_dir(JMiPOD)
        + fs.find_images_in_dir(JUNIWARD)
        + fs.find_images_in_dir(UERD)
    )
    # dataset = dataset[:500]

    mean, std = compute_mean_std(tqdm(dataset))
    print(mean.size())
    print(std.size())
    print("Mean", np.array2string(to_numpy(mean), precision=2, separator=",", max_line_width=119))
    print("Std ", np.array2string(to_numpy(std), precision=2, separator=",", max_line_width=119))


if __name__ == "__main__":
    main()
