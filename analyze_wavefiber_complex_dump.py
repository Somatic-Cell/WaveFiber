#!/usr/bin/env python3
"""
Analyze complex field dumps from the WaveFiber3d `treePO.cu` patch.

This script:
1. Reads the dumped Ex/Ey/Ez/Hx/Hy/Hz fields and outgoing directions.
2. Reconstructs the projected electric field used by `computeintensity`.
3. Verifies that reconstructed intensity matches the saved intensity dump.
4. Projects the field to a scalar complex component (E_theta / E_phi / dominant).
5. Builds a local demodulated complex residual.
6. Extracts band-limited complex coefficients with FFT radial windows.
7. Compares circular complex Gaussian, full 2D Gaussian, and PWNCG models.

Designed for the "kill test" stage: determine whether a flexible complex
distribution is meaningfully useful before modifying the practical fitting code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.special import gamma, hyp1f1


# -----------------------------
# IO helpers
# -----------------------------
def read_meta(path: str | Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, v = line.split(maxsplit=1)
            meta[k] = v
    return meta


def as_int(meta: Dict[str, str], key: str) -> int:
    return int(meta[key])


def as_float(meta: Dict[str, str], key: str) -> float:
    return float(meta[key])


def read_float_bin(path: str | Path, count: int | None = None) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if count is not None and arr.size != count:
        raise ValueError(f"{path}: expected {count} float32 values, found {arr.size}")
    return arr


def read_complex_float2_bin(path: str | Path, count: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2 != 0:
        raise ValueError(f"{path}: file length is not divisible by 2")
    arr = raw.reshape(-1, 2)
    if count is not None and arr.shape[0] != count:
        raise ValueError(f"{path}: expected {count} complex values, found {arr.shape[0]}")
    return arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)


# -----------------------------
# Electromagnetic reconstruction
# -----------------------------
def project_fields_like_computeintensity(
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray,
    Hx: np.ndarray,
    Hy: np.ndarray,
    Hz: np.ndarray,
    dirs: np.ndarray,
    Z0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dirx = dirs[:, 0]
    diry = dirs[:, 1]
    dirz = dirs[:, 2]

    dotE = Ex * dirx + Ey * diry + Ez * dirz
    Ex_t = Ex - dotE * dirx
    Ey_t = Ey - dotE * diry
    Ez_t = Ez - dotE * dirz

    dotH = Hx * dirx + Hy * diry + Hz * dirz
    Hx_t = Hx - dotH * dirx
    Hy_t = Hy - dotH * diry
    Hz_t = Hz - dotH * dirz

    Ex_p = Ex_t - (Hz_t * diry - Hy_t * dirz) * Z0
    Ey_p = Ey_t - (-(Hz_t * dirx - Hx_t * dirz)) * Z0
    Ez_p = Ez_t - (Hy_t * dirx - Hx_t * diry) * Z0

    intensity = np.abs(Ex_p) ** 2 + np.abs(Ey_p) ** 2 + np.abs(Ez_p) ** 2
    return Ex_p, Ey_p, Ez_p, intensity


def outgoing_spherical_basis(dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dirx = dirs[:, 0]
    diry = dirs[:, 1]
    dirz = dirs[:, 2]

    theta = np.arcsin(np.clip(dirz, -1.0, 1.0))
    phi = np.arctan2(diry, dirx)

    e_theta = np.stack(
        [-np.sin(theta) * np.cos(phi), -np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=1,
    )
    e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
    return e_theta, e_phi


def select_scalar_component(
    Ex_p: np.ndarray,
    Ey_p: np.ndarray,
    Ez_p: np.ndarray,
    dirs: np.ndarray,
    component: str,
) -> Tuple[np.ndarray, str, Dict[str, float]]:
    e_theta, e_phi = outgoing_spherical_basis(dirs)
    E_theta = Ex_p * e_theta[:, 0] + Ey_p * e_theta[:, 1] + Ez_p * e_theta[:, 2]
    E_phi = Ex_p * e_phi[:, 0] + Ey_p * e_phi[:, 1] + Ez_p * e_phi[:, 2]

    theta_energy = float(np.mean(np.abs(E_theta) ** 2))
    phi_energy = float(np.mean(np.abs(E_phi) ** 2))

    if component == "theta":
        return E_theta, "theta", {"theta_energy": theta_energy, "phi_energy": phi_energy}
    if component == "phi":
        return E_phi, "phi", {"theta_energy": theta_energy, "phi_energy": phi_energy}
    if component == "dominant":
        if theta_energy >= phi_energy:
            return E_theta, "theta", {"theta_energy": theta_energy, "phi_energy": phi_energy}
        return E_phi, "phi", {"theta_energy": theta_energy, "phi_energy": phi_energy}
    raise ValueError(f"Unknown component: {component}")


# -----------------------------
# Residual construction
# -----------------------------
def gaussian_filter_complex(arr: np.ndarray, sigma: float) -> np.ndarray:
    real = gaussian_filter(arr.real, sigma=sigma, mode=("nearest", "wrap"))
    imag = gaussian_filter(arr.imag, sigma=sigma, mode=("nearest", "wrap"))
    return real + 1j * imag


def compute_residual(
    field: np.ndarray,
    sigma: float,
    residual_kind: str,
    envelope_floor_frac: float,
    mask_percentile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_field = gaussian_filter_complex(field, sigma=sigma)
    envelope = np.abs(mean_field)
    floor = envelope_floor_frac * float(np.median(envelope))

    phase_demod = np.exp(-1j * np.angle(mean_field + 1e-30))
    if residual_kind == "multiplicative":
        residual = field * phase_demod / np.maximum(envelope, floor) - 1.0
    elif residual_kind == "additive":
        residual = field * phase_demod - envelope
    else:
        raise ValueError(f"Unknown residual kind: {residual_kind}")

    if mask_percentile > 0:
        cutoff = np.percentile(envelope, mask_percentile)
        mask = envelope >= cutoff
    else:
        mask = np.ones_like(envelope, dtype=bool)
    return residual, mask, mean_field


# -----------------------------
# Bandpass coefficients
# -----------------------------
def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def radial_gaussian_bandpass(
    field: np.ndarray,
    center: float,
    sigma_frac: float,
) -> np.ndarray:
    h, w = field.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    rr = np.sqrt((fy[:, None] / 0.5) ** 2 + (fx[None, :] / 0.5) ** 2)
    width = max(center * sigma_frac, 0.02)
    window = np.exp(-0.5 * ((rr - center) / width) ** 2)
    window[0, 0] = 0.0
    coeff = np.fft.ifft2(np.fft.fft2(field) * window)
    return coeff


# -----------------------------
# Statistical models
# -----------------------------
def laguerre_func(order: float, x: np.ndarray | float) -> np.ndarray | float:
    return hyp1f1(-order, 1.0, x)


def pwncg_logpdf(z: np.ndarray, mu: complex, sigma2: float, alpha: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128)
    if sigma2 <= 0 or alpha <= 0:
        return np.full(z.shape, -np.inf)
    eta = abs(mu) ** 2 / sigma2
    L = laguerre_func(alpha - 1.0, -eta)
    if np.real(L) <= 0 or not np.isfinite(L):
        return np.full(z.shape, -np.inf)
    return (
        -np.log(np.pi)
        - alpha * np.log(sigma2)
        - np.log(gamma(alpha))
        - np.log(L)
        + 2 * (alpha - 1.0) * np.log(np.maximum(np.abs(z), 1e-300))
        - np.abs(z - mu) ** 2 / sigma2
    )


def complex_gaussian_logpdf(z: np.ndarray, mu: complex, sigma2: float) -> np.ndarray:
    return -np.log(np.pi * sigma2) - np.abs(z - mu) ** 2 / sigma2


def real_gaussian_logpdf(z: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.real(z), np.imag(z)])
    inv = np.linalg.inv(cov)
    diff = X - mu
    maha = np.einsum("...i,ij,...j->...", diff, inv, diff)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.full(X.shape[0], -np.inf)
    return -0.5 * (2 * np.log(2 * np.pi) + logdet + maha)


def fit_complex_gaussian(z: np.ndarray) -> Dict[str, object]:
    mu = np.mean(z)
    sigma2 = float(np.mean(np.abs(z - mu) ** 2))
    return {"mu": mu, "sigma2": sigma2}


def fit_real_gaussian(z: np.ndarray) -> Dict[str, object]:
    X = np.column_stack([np.real(z), np.imag(z)])
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False, bias=True)
    cov = cov + 1e-8 * np.eye(2)
    return {"mu": mu, "cov": cov}


def pwncg_nll(params: np.ndarray, z: np.ndarray) -> float:
    mu = params[0] + 1j * params[1]
    sigma2 = float(np.exp(params[2]))
    alpha = float(np.exp(params[3]))
    lp = pwncg_logpdf(z, mu, sigma2, alpha)
    if not np.all(np.isfinite(lp)):
        return 1e100
    return -float(np.sum(lp))


def fit_pwncg(z: np.ndarray, n_starts: int = 6, seed: int = 0) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    mu0 = np.mean(z)
    sigma20 = float(np.mean(np.abs(z - mu0) ** 2))

    starts = [
        np.array([mu0.real, mu0.imag, np.log(max(sigma20, 1e-12)), np.log(1.0)]),
        np.array([mu0.real, mu0.imag, np.log(max(sigma20, 1e-12)), np.log(0.7)]),
        np.array([mu0.real, mu0.imag, np.log(max(sigma20, 1e-12)), np.log(1.5)]),
        np.array([mu0.real, mu0.imag, np.log(max(sigma20 * 0.7, 1e-12)), np.log(1.2)]),
        np.array([mu0.real, mu0.imag, np.log(max(sigma20 * 1.3, 1e-12)), np.log(0.8)]),
    ]
    while len(starts) < n_starts:
        starts.append(
            np.array(
                [
                    mu0.real + rng.normal(scale=np.std(np.real(z)) * 0.2 + 1e-6),
                    mu0.imag + rng.normal(scale=np.std(np.imag(z)) * 0.2 + 1e-6),
                    np.log(max(sigma20 * np.exp(rng.normal(scale=0.4)), 1e-12)),
                    np.log(np.exp(rng.normal(scale=0.6))),
                ]
            )
        )

    best = None
    for x0 in starts:
        res = minimize(
            pwncg_nll,
            x0,
            args=(z,),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
        )
        if best is None or res.fun < best.fun:
            best = res
    mu = best.x[0] + 1j * best.x[1]
    sigma2 = float(np.exp(best.x[2]))
    alpha = float(np.exp(best.x[3]))
    return {"mu": mu, "sigma2": sigma2, "alpha": alpha}


# -----------------------------
# Plotting
# -----------------------------
def scatter_with_contours(
    z: np.ndarray,
    cg: Dict[str, object],
    pw: Dict[str, object],
    outpath: str | Path,
    max_points: int = 5000,
) -> None:
    rng = np.random.default_rng(0)
    if z.size > max_points:
        z = z[rng.choice(z.size, size=max_points, replace=False)]

    xmin = np.percentile(np.real(z), 0.5)
    xmax = np.percentile(np.real(z), 99.5)
    ymin = np.percentile(np.imag(z), 0.5)
    ymax = np.percentile(np.imag(z), 99.5)
    pad_x = 0.15 * max(xmax - xmin, 1e-4)
    pad_y = 0.15 * max(ymax - ymin, 1e-4)
    xs = np.linspace(xmin - pad_x, xmax + pad_x, 220)
    ys = np.linspace(ymin - pad_y, ymax + pad_y, 220)
    X, Y = np.meshgrid(xs, ys)
    ZZ = X + 1j * Y
    LP_cg = complex_gaussian_logpdf(ZZ, cg["mu"], float(cg["sigma2"]))
    LP_pw = pwncg_logpdf(ZZ, pw["mu"], float(pw["sigma2"]), float(pw["alpha"]))

    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(z), np.imag(z), s=4, alpha=0.15)
    plt.contour(X, Y, np.exp(LP_cg), levels=6)
    plt.contour(X, Y, np.exp(LP_pw), levels=6, linestyles="dashed")
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Representative band coefficients")
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main analysis
# -----------------------------
def analyze_prefix(
    prefix: str | Path,
    outdir: str | Path,
    component: str,
    demod_sigma: float,
    residual_kind: str,
    envelope_floor_frac: float,
    mask_percentile: float,
    band_centers: List[float],
    band_sigma_frac: float,
    max_samples_per_band: int,
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    prefix = str(prefix)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = read_meta(prefix + "_meta.txt")
    thetaonum = as_int(meta, "thetaonum")
    phionum = as_int(meta, "phionum")
    obnum = as_int(meta, "obnum")
    Z0 = as_float(meta, "Z0")

    Ex = read_complex_float2_bin(prefix + "_Ex.bin", count=obnum)
    Ey = read_complex_float2_bin(prefix + "_Ey.bin", count=obnum)
    Ez = read_complex_float2_bin(prefix + "_Ez.bin", count=obnum)
    Hx = read_complex_float2_bin(prefix + "_Hx.bin", count=obnum)
    Hy = read_complex_float2_bin(prefix + "_Hy.bin", count=obnum)
    Hz = read_complex_float2_bin(prefix + "_Hz.bin", count=obnum)
    dirs = read_float_bin(prefix + "_dir.bin", count=obnum * 3).reshape(obnum, 3)
    saved_intensity = read_float_bin(prefix + "_intensity.bin", count=obnum)

    Ex_p, Ey_p, Ez_p, reconstructed_intensity = project_fields_like_computeintensity(
        Ex, Ey, Ez, Hx, Hy, Hz, dirs, Z0
    )

    abs_diff = np.abs(reconstructed_intensity - saved_intensity.astype(np.float64))
    intensity_summary = {
        "saved_mean": float(np.mean(saved_intensity)),
        "reconstructed_mean": float(np.mean(reconstructed_intensity)),
        "max_abs_error": float(np.max(abs_diff)),
        "mean_abs_error": float(np.mean(abs_diff)),
        "relative_l2_error": float(
            np.linalg.norm(reconstructed_intensity - saved_intensity)
            / (np.linalg.norm(saved_intensity) + 1e-30)
        ),
    }
    with open(outdir / "intensity_check.json", "w", encoding="utf-8") as f:
        json.dump(intensity_summary, f, indent=2)

    scalar, chosen_component, energy_summary = select_scalar_component(
        Ex_p, Ey_p, Ez_p, dirs, component
    )
    scalar_grid = scalar.reshape(thetaonum, phionum)

    residual, mask, mean_field = compute_residual(
        scalar_grid,
        sigma=demod_sigma,
        residual_kind=residual_kind,
        envelope_floor_frac=envelope_floor_frac,
        mask_percentile=mask_percentile,
    )

    # Save a quick diagnostic image of the local-mean envelope and the mask.
    envelope = np.abs(mean_field)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(np.log10(envelope + 1e-12), aspect="auto")
    axes[0].set_title("log10 local mean envelope")
    axes[1].imshow(mask.astype(float), aspect="auto")
    axes[1].set_title("coefficient mask")
    for ax in axes:
        ax.set_xlabel("phi index")
        ax.set_ylabel("theta index")
    fig.tight_layout()
    fig.savefig(outdir / "residual_mask_diagnostic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, object]] = []

    best_row = None
    best_coeffs = None
    for band_id, center in enumerate(band_centers):
        coeff = radial_gaussian_bandpass(residual, center=center, sigma_frac=band_sigma_frac)
        coeff_flat = coeff[mask]
        coeff_flat = coeff_flat[np.isfinite(coeff_flat)]
        if coeff_flat.size < 200:
            rows.append(
                {
                    "band_id": band_id,
                    "center_freq": center,
                    "num_coeffs": int(coeff_flat.size),
                    "status": "too_few_coeffs",
                }
            )
            continue

        if coeff_flat.size > max_samples_per_band:
            coeff_flat = coeff_flat[
                rng.choice(coeff_flat.size, size=max_samples_per_band, replace=False)
            ]

        perm = rng.permutation(coeff_flat.size)
        split = max(coeff_flat.size * 2 // 3, 1)
        train = coeff_flat[perm[:split]]
        test = coeff_flat[perm[split:]]
        if test.size < 100:
            rows.append(
                {
                    "band_id": band_id,
                    "center_freq": center,
                    "num_coeffs": int(coeff_flat.size),
                    "status": "too_few_test_coeffs",
                }
            )
            continue

        cg = fit_complex_gaussian(train)
        rg = fit_real_gaussian(train)
        pw = fit_pwncg(train, n_starts=6, seed=random_seed + band_id + 1)

        ll_cg = float(np.mean(complex_gaussian_logpdf(test, cg["mu"], float(cg["sigma2"]))))
        ll_rg = float(np.mean(real_gaussian_logpdf(test, rg["mu"], rg["cov"])))
        ll_pw = float(np.mean(pwncg_logpdf(test, pw["mu"], float(pw["sigma2"]), float(pw["alpha"]))))

        row = {
            "band_id": band_id,
            "center_freq": center,
            "num_coeffs": int(coeff_flat.size),
            "component": chosen_component,
            "residual_kind": residual_kind,
            "demod_sigma": demod_sigma,
            "mask_percentile": mask_percentile,
            "envelope_floor_frac": envelope_floor_frac,
            "ll_circular_cg": ll_cg,
            "ll_full_2d_gaussian": ll_rg,
            "ll_pwncg": ll_pw,
            "pwncg_minus_full_2d": ll_pw - ll_rg,
            "pwncg_minus_circular": ll_pw - ll_cg,
            "fitted_alpha": float(pw["alpha"]),
            "fitted_mu_real": float(np.real(pw["mu"])),
            "fitted_mu_imag": float(np.imag(pw["mu"])),
            "fitted_sigma2": float(pw["sigma2"]),
            "mean_abs_coeff": float(np.mean(np.abs(coeff_flat))),
            "status": "ok",
        }
        rows.append(row)

        if best_row is None or row["pwncg_minus_full_2d"] > best_row["pwncg_minus_full_2d"]:
            best_row = row
            best_coeffs = coeff_flat
            best_cg = cg
            best_pw = pw

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "band_summary.csv", index=False)

    summary = {
        "prefix": prefix,
        "chosen_component": chosen_component,
        "energy_summary": energy_summary,
        "intensity_summary": intensity_summary,
        "analysis_params": {
            "component": component,
            "demod_sigma": demod_sigma,
            "residual_kind": residual_kind,
            "envelope_floor_frac": envelope_floor_frac,
            "mask_percentile": mask_percentile,
            "band_centers": band_centers,
            "band_sigma_frac": band_sigma_frac,
            "max_samples_per_band": max_samples_per_band,
            "random_seed": random_seed,
        },
    }
    with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best_coeffs is not None:
        scatter_with_contours(best_coeffs, best_cg, best_pw, outdir / "representative_band_scatter.png")

    return df, summary


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True, help="Path prefix written by the treePO patch, without the *_Ex.bin suffix.")
    p.add_argument("--outdir", default=None, help="Directory for analysis outputs. Defaults to <prefix>_analysis.")
    p.add_argument("--component", choices=["theta", "phi", "dominant"], default="dominant")
    p.add_argument("--demod-sigma", type=float, default=6.0)
    p.add_argument("--residual", choices=["multiplicative", "additive"], default="multiplicative")
    p.add_argument("--envelope-floor-frac", type=float, default=0.10)
    p.add_argument("--mask-percentile", type=float, default=25.0)
    p.add_argument("--band-centers", default="0.05,0.10,0.18,0.30")
    p.add_argument("--band-sigma-frac", type=float, default=0.50)
    p.add_argument("--max-samples-per-band", type=int, default=30000)
    p.add_argument("--random-seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argument_parser().parse_args()
    prefix = args.prefix
    outdir = args.outdir or (prefix + "_analysis")
    df, summary = analyze_prefix(
        prefix=prefix,
        outdir=outdir,
        component=args.component,
        demod_sigma=args.demod_sigma,
        residual_kind=args.residual,
        envelope_floor_frac=args.envelope_floor_frac,
        mask_percentile=args.mask_percentile,
        band_centers=parse_float_list(args.band_centers),
        band_sigma_frac=args.band_sigma_frac,
        max_samples_per_band=args.max_samples_per_band,
        random_seed=args.random_seed,
    )
    print(df)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
