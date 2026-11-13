"""Build and install the distribution, then use its data outside the checkout.

Requires the project dependencies, build, setuptools>=77 and wheel.
Run with: python -m unittest discover -s tests -v
"""

from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


RESOURCES = (
    "Cook23.dat", "Cook23bin.dat", "Dinnbier22.dat", "Y24Martin19.dat",
    "Yan2024.dat", "Yan24all_xi.py", "bastian08.dat", "larsen02.dat",
    "mmaxMecl.dat", "randriamanakoto13.dat", "vanDokkumConroy24.dat",
    "vanDokkumConroy24range.dat",
)

SMOKE = r'''
from pathlib import Path
import sys

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import igimf
assert Path(igimf.__file__).resolve().is_relative_to(target), igimf.__file__

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from igimf import utilities
from igimf.plots import Plots

martin = utilities.Y24Martin19alpha1()
yan = utilities.import_Yan24all_xi()
assert martin.shape == (18, 4), martin.shape
assert yan.shape == (561, 5), yan.shape
# Reference results from the original repository implementation.
np.testing.assert_allclose(martin["alpha1"].iloc[:3],
    [1.0837339375265902, 1.1382109784712098, 1.1986275763369574], rtol=1e-12)
np.testing.assert_allclose(yan["alpha1"].iloc[:3],
    [0.9613927523536148, 1.177443156120408, 1.4164551453967644], rtol=1e-12)
assert np.isfinite(martin[["Z", "alpha1"]].to_numpy()).all()
assert np.isfinite(yan[["Z", "alpha1"]].to_numpy()).all()

assert not Path("data").exists()
assert not Path("figs").exists()
plots = Plots()
plots.Cook23_plot()
plots.Meclmax_vs_SFR_observations(np.array([0.1, 1.0]), np.array([1e3, 1e4]))
plots.alpha1_Z_plot()
for name in ("Cook23.pdf", "Meclmax_vs_SFR_observations.pdf", "alpha1_alternative.pdf"):
    assert (Path("figs") / name).read_bytes().startswith(b"%PDF"), name
plt.close("all")

# Missing scientific inputs must be reported rather than substituted or skipped.
for kwargs in ({}, {"data_dir": Path.cwd()}):
    try:
        plots.Fig11_plot(**kwargs)
    except FileNotFoundError as error:
        message = str(error)
        for name in ("Capuzzo-dolcetta17CMOl.csv", "Capuzzo-dolcetta17CMOu.csv",
                     "Capuzzo-dolcetta17BH.csv"):
            assert name in message, message
        assert "data_dir" in message, message
    else:
        raise AssertionError("Fig11_plot accepted missing reference datasets")
assert not (Path("figs") / "Fig11.pdf").exists()
'''


class InstalledDataSmokeTest(unittest.TestCase):
    def test_installed_data_outside_repository(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, dist, target, outside = [root / name for name in
                                             ("source", "dist", "installed", "outside")]
            shutil.copytree(repo, source, ignore=shutil.ignore_patterns(
                ".git", ".venv", "venv", "build", "dist", "*.egg-info", "__pycache__",
                ".pytest_cache", "figs",
            ))
            outside.mkdir()

            def run(*args, cwd):
                result = subprocess.run(
                    [sys.executable, *map(str, args)], cwd=cwd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout)

            # build's default builds a wheel from the sdist, testing both artifacts.
            run("-m", "build", "--no-isolation", "--outdir", dist, source, cwd=outside)
            wheel, = dist.glob("*.whl")
            sdist, = dist.glob("*.tar.gz")
            with zipfile.ZipFile(wheel) as whl, tarfile.open(sdist) as src:
                prefix = src.getnames()[0].split("/")[0]
                for name in RESOURCES:
                    relative = f"igimf/data/{name}"
                    expected = (repo / relative).read_bytes()
                    self.assertEqual(whl.read(relative), expected, relative)
                    with src.extractfile(f"{prefix}/{relative}") as stream:
                        self.assertEqual(stream.read(), expected, relative)

            run("-m", "pip", "install", "--no-deps", "--no-index", "--target", target,
                wheel, cwd=outside)
            # -I excludes the checkout, PYTHONPATH and the user site from imports.
            run("-I", "-c", SMOKE, target, cwd=outside)


if __name__ == "__main__":
    unittest.main()
