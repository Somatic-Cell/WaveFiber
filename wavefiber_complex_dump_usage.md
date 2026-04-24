# WaveFiber3d complex dump patch + analysis

This bundle contains:

- `wavefiber_complex_dump.patch`: minimal patch for `fibersim/treePO.cu`
- `analyze_wavefiber_complex_dump.py`: external analysis script
- `wavefiber_complex_dump_usage.md`: this file

## What the patch does

It keeps the original intensity output unchanged and additionally writes, for each run:

- `<prefix>_meta.txt`
- `<prefix>_intensity.bin`
- `<prefix>_dir.bin`
- `<prefix>_Ex.bin`, `_Ey.bin`, `_Ez.bin`
- `<prefix>_Hx.bin`, `_Hy.bin`, `_Hz.bin`

The new prefix is

```text
output/fiber_<fiberindex>_lambda<lambdaindex>_ti<thetaiindex>_pi<phiiindex>_T<mode>_depth<max_depth>_s<startindex>_e<endindex>
```

Binary formats:

- `*_intensity.bin`: `float32`, length `obnum`
- `*_dir.bin`: `float32`, length `3 * obnum`, interleaved xyz
- `*_Ex.bin` etc.: `float32`, length `2 * obnum`, interleaved real/imag
- `*_meta.txt`: key-value text

## How to apply

From the WaveFiber3d repo root:

```bash
git apply /path/to/wavefiber_complex_dump.patch
```

If `git apply` complains because your local file has drifted, open `fibersim/treePO.cu` and manually apply the same changes.

## How to build

From `fibersim/`:

```bash
nvcc treePO.cu -rdc=true -o treePO -lcufft
```

That is the compile command given in the repo README.

## How to run

Run `treePO` exactly as before. After a run, look in `output/` for files matching the new prefix.

Example analysis command:

```bash
python /path/to/analyze_wavefiber_complex_dump.py \
  --prefix output/fiber_0_lambda0_ti90_pi0_TM_depth3_s0_e2999 \
  --component dominant \
  --demod-sigma 6 \
  --residual multiplicative \
  --envelope-floor-frac 0.10 \
  --mask-percentile 25 \
  --outdir output/analysis_f0_l0_ti90_pi0
```

## Recommended first experiment

Start small.

- Use `fiberindex = 0` or `1`, not `2` or `3`.
- Use one wavelength.
- Use one incident theta / phi.
- Keep the default practical-fidelity settings otherwise.
- Verify `intensity_check.json` first.

## What the analysis script does

1. Reconstructs the projected electric field used by `computeintensity`
2. Verifies dumped fields against saved intensity
3. Projects onto `E_theta`, `E_phi`, or the dominant component
4. Builds a local complex residual by demodulating with a smoothed local mean
5. Extracts band-limited complex coefficients using FFT radial windows
6. Fits and compares:
   - circular complex Gaussian
   - full 2D Gaussian on `(Re, Im)`
   - PWNCG

Outputs include:

- `band_summary.csv`
- `intensity_check.json`
- `run_summary.json`
- `residual_mask_diagnostic.png`
- `representative_band_scatter.png`

## First thing to look for

In `band_summary.csv`, focus on:

- `pwncg_minus_full_2d`
- `fitted_alpha`

If `pwncg_minus_full_2d` is consistently near zero after demodulation and masking, that is evidence **against** the need for a more flexible complex distribution in this pipeline.

If it remains clearly positive on glint-dominant bands, this direction stays alive.

## Important caveat

This script is a **kill test**, not a final renderer evaluation.  
It answers: “Does a flexible complex distribution still matter after local demodulation and band extraction?”

It does **not** yet model cross-band dependence, spatial copulas, or final appearance synthesis.
