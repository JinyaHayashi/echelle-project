"""
Raman spectrum processing from echelle imaging data.

This module:
    1. Loads calibration data
    2. Computes order spectra
    3. Performs background subtraction
    4. Applies sensitivity correction
    5. Concatenates wavelength regions
    6. Outputs Raman shift vs intensity

Required external classes:
    - Calibrations
    - EchelleImage
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from echelle_pkg.echelle_spectra import Calibrations, EchelleImage
from echelle_pkg.gragh_tools import ticks_visual, grid_visual


# ============================================================
# Calibration setup
# ============================================================

def setup_calibration(calibration_path, files_cmos, spec="fujii"):
    cb = Calibrations(
        calibration_path,
        files_cmos,
        spec=spec,
        crop=[100, 1850],
        crop2=[20, 1095],
    )
    cb.load_pattern()
    cb.load_sphere()
    cb.make_cutting_masks()
    return cb


# ============================================================
# Order spectrum calculation
# ============================================================

def compute_order_spectra(image_dir, file_base, ext, frange, calibration):
    spectra_list = []

    for i in frange:
        fname = f"{file_base}{i:02d}{ext}"
        fpath = os.path.join(image_dir, fname)

        img = EchelleImage(
            fpath,
            clbr=calibration,
            spec="fujii",
            crop=[100, 1850],
            crop2=[20, 1095],
        )
        img.calculate_order_spectra()
        spectra_list.append(img.order_spectra[0])

    return spectra_list


def sum_spectra(spectra_list):
    total = np.zeros_like(spectra_list[0])
    for sp in spectra_list:
        total += sp
    return total


# ============================================================
# Sensitivity correction
# ============================================================

def load_sensitivity_data(csv_path):
    df = pd.read_csv(csv_path)
    grouped = df.groupby("order")

    order_sdata = {}
    for order, group in grouped:
        x = group["wavelength"].to_numpy()
        y = group["sensitivity"].to_numpy()
        order_sdata[order] = (x, y)

    return order_sdata


def apply_sensitivity_correction(raman_sum_list, order_sdata, stack_number=18):
    calibrated = []

    for i in range(14):  # order 10?23
        value = (
            order_sdata[i][1] * raman_sum_list[i + 10]
        ) / stack_number
        calibrated.append(value)

    return calibrated


# ============================================================
# Wavelength stitching
# ============================================================

def stitch_wavelength(order_sdata):
    wavelength = np.concatenate((
        order_sdata[0][0][171:1749],
        order_sdata[1][0][188:1431],
        order_sdata[2][0][202:1420],
        order_sdata[3][0][214:1413],
        order_sdata[4][0][223:1401],
        order_sdata[5][0][236:1389],
        order_sdata[6][0][274:1387],
        order_sdata[7][0][273:1371],
        order_sdata[8][0][298:1356],
        order_sdata[9][0][315:1331],
        order_sdata[10][0][275:1307],
        order_sdata[11][0][245:1378],
        order_sdata[12][0][333:1418],
        order_sdata[13][0][0:1357],
    ))
    return wavelength


def stitch_intensity(calibrated_raman):
    intensity = np.concatenate((
        calibrated_raman[0][171:1749],
        calibrated_raman[1][188:1431],
        calibrated_raman[2][202:1420],
        calibrated_raman[3][214:1413],
        calibrated_raman[4][223:1401],
        calibrated_raman[5][236:1389],
        calibrated_raman[6][274:1387],
        calibrated_raman[7][273:1371],
        calibrated_raman[8][298:1356],
        calibrated_raman[9][315:1331],
        calibrated_raman[10][275:1307],
        calibrated_raman[11][245:1378],
        calibrated_raman[12][333:1418],
        calibrated_raman[13][0:1357],
    ))
    return intensity


# ============================================================
# Raman shift conversion
# ============================================================

def convert_to_raman_shift(wavelength, laser_nm=532):
    return (1 / laser_nm - 1 / wavelength) * 1e7


# ============================================================
# Plotting
# ============================================================

def plot_raman(raman_shift, intensity):
    fig = plt.figure(figsize=(4, 2.2), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(raman_shift, intensity, s=0.1, c="k")

    rcParams['font.family'] = 'Times New Roman'
    rcParams["font.size"] = 10

    ax.set_xlabel("Raman shift [$\mathrm{cm^{-1}}$]")
    ax.set_ylabel("Intensity [arb. units]")
    ax.set_xlim(50, 2000)

    ticks_visual(ax, l1=4, l2=2)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main execution
# ============================================================

def main():

    # --------------------------
    # User settings
    # --------------------------

    path = "BoronFilm_Raman_data"
    data_dir = os.path.join(path, "20251202_deposited-boronfilm_10s")
    bg_dir = os.path.join(path, "20251202_deposited-boronfilm-back_10s")

    fb = "20251202_deposited-boronfilm_10s_"
    fb_bg = "20251202_deposited-boronfilm-back_10s_"
    ext = ".tif"
    frange = range(0, 17)

    files_cmos = {
        "orders": "pattern.txt",
        "sphr": "20250123_sphere_100ms.tif",
        "bkgr": "20250123_sphereback_100ms.tif",
    }

    calibration_path = "../examples/calibration_file_examples"

    # --------------------------
    # Processing pipeline
    # --------------------------

    cb = setup_calibration(calibration_path, files_cmos)

    raman_list = compute_order_spectra(data_dir, fb, ext, frange, cb)
    raman_bg_list = compute_order_spectra(bg_dir, fb_bg, ext, frange, cb)

    raman_sum = sum_spectra(raman_list)
    raman_bg_sum = sum_spectra(raman_bg_list)

    raman_subtracted = raman_sum - raman_bg_sum

    order_sdata = load_sensitivity_data("sensitivity_data.csv")

    calibrated = apply_sensitivity_correction(
        raman_subtracted,
        order_sdata,
    )

    wavelength = stitch_wavelength(order_sdata)
    intensity = stitch_intensity(calibrated)

    raman_shift = convert_to_raman_shift(wavelength)

    plot_raman(raman_shift, intensity)


if __name__ == "__main__":
    main()
