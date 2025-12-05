#!/usr/bin/env python
"""
gaia_accel_gen_fit.py

Generate a 1-star HGCA-like FITS file for a Gaia DR3
acceleration solution (nss_acceleration_astro), using
values provided in gaia_accel_prep.ini.

Usage:
    python gaia_accel_gen_fit.py gaia_accel_prep.ini

In your orvara config:
    HipID   = hip_id from [star]
    HGCAFile = OUTPUT_FITS from [paths]
    use_epoch_astrometry = False
"""

import argparse
import configparser
import re
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from astropy.io import fits

GAIA_REF_EPOCH = 2016.0  # DR3 reference epoch (J2016)


# ==================== CONFIG STRUCT ====================

@dataclass
class PrepConfig:
    hgca_template: str
    output_fits: str

    source_id: int
    hip_id: int

    npar: int

    ra: float
    ra_err: float
    dec: float
    dec_err: float
    plx: float
    plx_err: float
    pmra: float
    pmra_err: float
    pmdec: float
    pmdec_err: float
    accra: float
    accra_err: float
    accdec: float
    accdec_err: float

    rv: float
    rv_err: float
    rv_source: str

    corr_vec: np.ndarray
    error_inflation_factor: float


def _get_float(sec: configparser.SectionProxy, key: str, default=np.nan) -> float:
    if key not in sec or sec.get(key) == "":
        return default
    return float(sec.get(key))


def parse_corr_vec(text: str, npar: int) -> np.ndarray:
    """
    Parse CORR_VEC string from config.

    Gaia docs: corr_vec is the upper triangle of the correlation
    matrix in column-major storage: for each column j>i, we store
    corr(i,j) for i=0..j-1, columns j=1..n-1. :contentReference[oaicite:2]{index=2}

    The archive table always has length 98, but only the first
    S = npar*(npar-1)/2 entries are relevant for a given solution.
    We allow the user to paste either S entries or the full 98;
    in the latter case we truncate to S.
    """
    if text is None or text.strip() == "":
        raise ValueError("corr_vec must be provided in [gaia_solution].")

    tokens = re.split(r"[,\s]+", text.strip())
    vals = [float(t) for t in tokens if t]
    vec = np.array(vals, dtype=float)

    needed = npar * (npar - 1) // 2
    if vec.size < needed:
        raise ValueError(
            f"corr_vec has {vec.size} entries; at least {needed} required for npar={npar}."
        )
    if vec.size > needed:
        vec = vec[:needed]
    return vec


def load_prep_config(path: str) -> PrepConfig:
    cp = configparser.ConfigParser()
    cp.read(path)

    paths = cp["paths"]
    star = cp["star"]
    g = cp["gaia_solution"]

    npar = int(g.get("npar"))

    corr_vec = parse_corr_vec(g.get("corr_vec"), npar=npar)

    # Error inflation factor (optional)
    if "advanced" in cp and "error_inflation_factor" in cp["advanced"]:
        error_factor = float(cp["advanced"].get("error_inflation_factor"))
    else:
        error_factor = 2.0  # default

    return PrepConfig(
        hgca_template=paths.get("HGCA_TEMPLATE"),
        output_fits=paths.get("OUTPUT_FITS"),

        source_id=int(star.get("source_id")),
        hip_id=int(star.get("hip_id")),

        npar=npar,
        ra=_get_float(g, "ra"),
        ra_err=_get_float(g, "ra_err"),
        dec=_get_float(g, "dec"),
        dec_err=_get_float(g, "dec_err"),
        plx=_get_float(g, "parallax"),
        plx_err=_get_float(g, "parallax_err"),
        pmra=_get_float(g, "pmra"),
        pmra_err=_get_float(g, "pmra_err"),
        pmdec=_get_float(g, "pmdec"),
        pmdec_err=_get_float(g, "pmdec_err"),
        accra=_get_float(g, "accra"),
        accra_err=_get_float(g, "accra_err"),
        accdec=_get_float(g, "accdec"),
        accdec_err=_get_float(g, "accdec_err"),
        rv=_get_float(g, "rv", default=np.nan),
        rv_err=_get_float(g, "rv_err", default=np.nan),
        rv_source=g.get("rv_source", ""),
        corr_vec=corr_vec,
        error_inflation_factor=error_factor,
    )


# ==================== CORRELATION & ROW BUILDING ====================

REQUIRED_NAMES = ["ra", "dec", "pmra", "pmdec", "accra", "accdec"]


def corr_vec_to_matrix(vec: np.ndarray, npar: int) -> np.ndarray:
    """
    Build full npar x npar correlation matrix from CORR_VEC,
    using Gaia's column-major upper-triangle storage order.

    For j in 1..npar-1:
        for i in 0..j-1:
            M[i,j] = M[j,i] = vec[k]; k++
    """
    needed = npar * (npar - 1) // 2
    if vec.size != needed:
        raise ValueError(
            f"corr_vec size {vec.size} inconsistent with npar={npar} "
            f"(expected {needed})."
        )

    M = np.eye(npar, dtype=float)
    k = 0
    for j in range(1, npar):
        for i in range(j):
            M[i, j] = M[j, i] = vec[k]
            k += 1
    return M


def build_row(cfg: PrepConfig) -> Dict[str, Any]:
    # Parameter ordering must follow Gaia docs:
    # Acceleration7: ra, dec, parallax, pmra, pmdec, accel_ra, accel_dec
    # Acceleration9: above + deriv_accel_ra, deriv_accel_dec
    if cfg.npar == 7:
        param_names = [
            "ra",
            "dec",
            "parallax",
            "pmra",
            "pmdec",
            "accra",
            "accdec",
        ]
    elif cfg.npar == 9:
        param_names = [
            "ra",
            "dec",
            "parallax",
            "pmra",
            "pmdec",
            "accra",
            "accdec",
            "deriv_accra",
            "deriv_accdec",
        ]
    else:
        raise ValueError(f"Unsupported npar={cfg.npar}; expected 7 or 9.")

    C = corr_vec_to_matrix(cfg.corr_vec, cfg.npar)
    idx = {name: i for i, name in enumerate(param_names)}

    for req in REQUIRED_NAMES:
        if req not in idx:
            raise KeyError(f"Required parameter '{req}' missing in param_names.")

    ra_pmra_corr = C[idx["ra"], idx["pmra"]]
    dec_pmdec_corr = C[idx["dec"], idx["pmdec"]]
    accra_accdec = C[idx["accra"], idx["accdec"]]
    pmra_pmdec = C[idx["pmra"], idx["pmdec"]]

    epoch_ra_gaia = GAIA_REF_EPOCH - ra_pmra_corr * cfg.ra_err / cfg.pmra_err
    epoch_dec_gaia = GAIA_REF_EPOCH - dec_pmdec_corr * cfg.dec_err / cfg.pmdec_err

    # Apply user-defined inflation to Gaia acceleration errors
    accra_err = cfg.accra_err * cfg.error_inflation_factor
    accdec_err = cfg.accdec_err * cfg.error_inflation_factor

    pmra = cfg.pmra
    pmdec = cfg.pmdec
    big = 1e7  # huge errors for synthetic Hip/HG proper motions

    row: Dict[str, Any] = dict(
        hip_id=int(cfg.hip_id),
        gaia_source_id=int(cfg.source_id),

        gaia_ra=cfg.ra,
        gaia_dec=cfg.dec,

        radial_velocity=cfg.rv,
        radial_velocity_error=cfg.rv_err,
        radial_velocity_source=cfg.rv_source,

        parallax_gaia=cfg.plx,
        parallax_gaia_error=cfg.plx_err,

        pmra_gaia=pmra,
        pmdec_gaia=pmdec,
        pmra_gaia_error=cfg.pmra_err,
        pmdec_gaia_error=cfg.pmdec_err,
        pmra_pmdec_gaia=pmra_pmdec,

        pmra_hg=pmra,
        pmdec_hg=pmdec,
        pmra_hg_error=big,
        pmdec_hg_error=big,
        pmra_pmdec_hg=0.0,

        pmra_hip=pmra,
        pmdec_hip=pmdec,
        pmra_hip_error=big,
        pmdec_hip_error=big,
        pmra_pmdec_hip=0.0,

        epoch_ra_gaia=float(epoch_ra_gaia),
        epoch_dec_gaia=float(epoch_dec_gaia),
        epoch_ra_hip=1991.0,
        epoch_dec_hip=1991.0,

        crosscal_pmra_hip=pmra,
        crosscal_pmdec_hip=pmdec,
        crosscal_pmra_hg=pmra,
        crosscal_pmdec_hg=pmdec,

        nonlinear_dpmra=pmra * 1.0e-4,
        nonlinear_dpmdec=pmdec * 1.0e-4,

        gaia_npar=cfg.npar,

        accra_gaia=cfg.accra,
        accdec_gaia=cfg.accdec,
        accra_gaia_error=accra_err,
        accdec_gaia_error=accdec_err,
        accra_accdec_gaia=accra_accdec,
    )

    return row


# ==================== FITS WRITER ====================

def write_hgca_like_fits(cfg: PrepConfig, row: Dict[str, Any]) -> None:
    """
    Use HGCA_vEDR3.fits as a template and produce a 1-row
    table with all columns needed by orvara, plus a few
    Gaia-specific extras (gaia_npar, accra_gaia, etc.).
    """
    base_hdul = fits.open(cfg.hgca_template)
    base_cols = base_hdul[1].columns
    base_header = base_hdul[1].header
    base_names = list(base_cols.names)

    extra = []

    def add_col(name: str, fmt: str):
        if name not in base_names:
            extra.append(fits.Column(name=name, format=fmt))

    add_col("gaia_npar", "J")
    add_col("gaia_ra", "D")
    add_col("gaia_dec", "D")
    add_col("gaia_source_id", "K")
    add_col("accra_gaia", "E")
    add_col("accdec_gaia", "E")
    add_col("accra_gaia_error", "E")
    add_col("accdec_gaia_error", "E")
    add_col("accra_accdec_gaia", "E")

    all_cols = base_cols + fits.ColDefs(extra)
    all_names = list(all_cols.names)

    arrays: Dict[str, np.ndarray] = {}
    for name in all_names:
        col = all_cols[name]
        fmt = col.format
        if name in row:
            arrays[name] = np.array([row[name]])
        else:
            if fmt.startswith("A"):
                arrays[name] = np.array([""])
            elif fmt[0] in ("J", "K", "I", "B", "L"):
                arrays[name] = np.zeros(1, dtype=np.int64)
            else:
                arrays[name] = np.full(1, np.nan, dtype=float)

    cols = [
        fits.Column(name=n, format=all_cols[n].format, array=arrays[n])
        for n in all_names
    ]
    table_hdu = fits.BinTableHDU.from_columns(cols, header=base_header)
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(
        cfg.output_fits, overwrite=True
    )


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description="Generate one-star HGCA-like Gaia acceleration FITS for orvara."
    )
    parser.add_argument("prep_ini", help="Path to gaia_accel_prep.ini")
    args = parser.parse_args()

    cfg = load_prep_config(args.prep_ini)
    row = build_row(cfg)
    write_hgca_like_fits(cfg, row)


if __name__ == "__main__":
    main()
