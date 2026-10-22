"""Generate calibrated OPM camera comparisons and their labeled PSNR figures.

All acquisition, specimen, noise and reconstruction settings come from --config.
One physical forward_model produces each raw camera realization shared by RL
and RLGC. --render-only reuses final numerical panels without reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse import csr_matrix
from skimage.metrics import peak_signal_noise_ratio

from opm_processing.imageprocessing.camera import camera_correct
from opm_processing.imageprocessing.opmtools import orthogonal_deskew
from tests.simulation.camera import Camera, digitize
from tests.simulation.forward_model import (
    Acquisition,
    Microscope,
    camera_psf,
    forward_model,
)
from tests.simulation.scenes import acquisition_shape, prepare_scene

Array = np.ndarray

METHODS = ("gt", "rl_best", "rl_end", "gc")
NAMES = {"gt": "GT", "rl_best": "RL best recorded", "rl_end": "RL end", "gc": "RLGC"}
COLORS = {"gt": "#111111", "rl_best": "#0072B2", "rl_end": "#E69F00", "gc": "#009E73"}
STROKE = [pe.withStroke(linewidth=1.7, foreground="black")]
VIEWS = {"XY": (0, ("y", "x")), "XZ": (1, ("z", "x")), "YZ": (2, ("z", "y"))}
LABELS = []


def laboratory(volume: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Apply production orthogonal deskew and rotation at camera-pixel pitch."""
    p = P.pixel_size_um
    result = orthogonal_deskew(
        volume,
        theta=P.angle_deg,
        distance=P.scan_step_um,
        pixel_size=p,
        downsample_factor=1,
        divisible_by=1,
    )
    half = (np.asarray(volume.shape) - 1) / 2
    theta = np.deg2rad(P.angle_deg)
    origin = (
        -half[1] * p * np.sin(theta),
        -half[0] * P.scan_step_um - half[1] * p * np.cos(theta),
        -half[2] * p,
    )
    axes = tuple(
        o + np.arange(n) * p for o, n in zip(origin, result.shape, strict=True)
    )
    return result, axes


def project(volume: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Crop physically aligned maximum-intensity projections."""
    return {
        v: volume.max(axis=axis)[np.ix_(*(masks[n] for n in names))]
        for v, (axis, names) in VIEWS.items()
    }


def generate_case(
    raw: np.ndarray,
    gt: dict[str, np.ndarray],
    axes: tuple[np.ndarray, ...],
    masks: dict[str, np.ndarray],
    psf: np.ndarray,
    factor: int,
    folder: Path,
    kind: str,
) -> None:
    """Run the sole deconvolver and retain final figure panels, not iterates."""
    import cupy as cp

    from opm_processing.imageprocessing.rlgc import rlgc

    observed = camera_correct(
        raw[::factor], P.camera.offset_adu, P.camera.electrons_per_adu
    )
    shape = ((observed.shape[0] - 1) * factor + 1, *observed.shape[1:])
    chosen = CONFIG["rendering"]["best_view"][kind]
    best = dict.fromkeys(VIEWS, (-np.inf, None, None))
    selected = None
    endpoint = None

    def record(step: int, device: Array) -> None:
        """Retain the requested best checkpoint and final reconstruction."""
        nonlocal selected, endpoint
        if step not in CONFIG["rl"]["checkpoints"]:
            return
        images = project(laboratory(cp.asnumpy(device))[0], masks)
        for view in VIEWS:
            score = peak_signal_noise_ratio(
                gt[view], images[view], data_range=float(np.max(gt[view]))
            )
            if score > best[view][0]:
                best[view] = (score, step, images[view])
                if view == chosen:
                    selected = images
        if step == RL_ITERATIONS:
            endpoint = images

    rlgc(
        observed,
        psf,
        scan_upsample_factor=factor,
        boundary="zero",
        gradient_consensus=False,
        max_iterations=RL_ITERATIONS,
        max_delta=0,
        initial_estimate=np.ones(shape, np.float32),
        callback=record,
        release_memory=False,
    )

    class StopLog(logging.Logger):
        def info(self, message: str, *args: object) -> None:
            """Capture the reference stopping reason and iteration count."""
            if "stop=" in message:
                self.reason, self.iterations = args[-2:]

    log = StopLog("figure-deconvolution", level=logging.INFO)
    result = rlgc(
        observed,
        psf,
        scan_upsample_factor=factor,
        boundary="zero",
        logger=log,
        release_memory=False,
        **CONFIG["rlgc"],
    )
    gc = project(laboratory(result)[0], masks)
    arrays = {n: a[masks[n]] for n, a in zip(("z", "y", "x"), axes, strict=True)}
    arrays.update({"gt_" + v: a for v, a in gt.items()})
    for method, images in (("rl_best", selected), ("rl_end", endpoint), ("gc", gc)):
        arrays.update({method + "_" + v: a for v, a in images.items()})
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "views.npz", **arrays)
    info = {
        "best_iteration": best[chosen][1],
        "gc_iteration": log.iterations,
        "gc_stop": log.reason,
        "best_view": chosen,
    }
    (folder / "views.json").write_text(json.dumps(info, indent=2))
    if kind == "line-pairs":
        profiles = {}
        metadata = []
        for view in ("XY", "XZ"):
            vertical = "y" if view == "XY" else "z"
            cut = int(np.argmin(abs(arrays[vertical])))
            profiles[view + "_x"] = arrays["x"]
            images = {
                "gt": gt[view],
                "rl_best": best[view][2],
                "rl_end": endpoint[view],
                "gc": gc[view],
            }
            for method, image in images.items():
                profiles[view + "_" + method] = image[cut]
            metadata.append(
                {
                    "view": view,
                    "best_iteration": best[view][1],
                    "gc_iteration": log.iterations,
                    "cut_coordinate_um": float(arrays[vertical][cut]),
                    "scores": {
                        m: peak_signal_noise_ratio(
                            gt[view], a, data_range=float(np.max(gt[view]))
                        )
                        for m, a in images.items()
                        if m != "gt"
                    },
                }
            )
        np.savez_compressed(folder / "profiles.npz", **profiles)
        (folder / "profiles.json").write_text(json.dumps(metadata, indent=2))


def simulate() -> None:
    """Use the shared fine-grid optical model; no alternate image simulator."""
    psf, scale = camera_psf(P)
    for profile in CONFIG["illumination_profiles"]:
        for kind in CONFIG["kinds"]:
            scene = None
            for photons, background in CONFIG["photon_budgets"]:
                folders = [
                    ROOT / f"photons_{photons}/{profile}_{kind}_f{factor}"
                    for factor in CONFIG["scan_factors"]
                ]
                marker = "profiles.npz" if kind == "line-pairs" else "views.npz"
                if all((folder / marker).exists() for folder in folders):
                    continue
                if scene is None:
                    scene = prepare_scene(
                        kind,
                        P,
                        parameters=CONFIG["specimens"][kind],
                        illumination_profile=profile,
                    )
                shape = acquisition_shape(scene)
                _, axes = laboratory(np.zeros(shape, np.float32))
                masks = {
                    n: (a >= fine[0]) & (a <= fine[-1])
                    for n, a, fine in zip(
                        ("z", "y", "x"), axes, scene.axes, strict=True
                    )
                }
                emission = forward_model(
                    scene.density,
                    Acquisition(
                        scene.axes,
                        shape,
                        illumination_profile=profile,
                        camera_photon_budget=(photons, background),
                        output="object",
                    ),
                    P,
                )
                truth = emission
                for axis, (fine, coarse) in enumerate(
                    zip(scene.axes, axes, strict=True)
                ):
                    overlap = np.maximum(
                        np.minimum(
                            coarse[:, None] + P.pixel_size_um / 2,
                            fine[None, :] + P.fine_spacing_um / 2,
                        )
                        - np.maximum(
                            coarse[:, None] - P.pixel_size_um / 2,
                            fine[None, :] - P.fine_spacing_um / 2,
                        ),
                        0,
                    )
                    moved = np.moveaxis(truth, axis, 0)
                    reduced = csr_matrix(overlap) @ moved.reshape(moved.shape[0], -1)
                    truth = np.moveaxis(
                        reduced.reshape((len(coarse), *moved.shape[1:])), 0, axis
                    )
                truth *= scale / (
                    P.scan_step_um
                    * P.pixel_size_um**2
                    * np.sin(np.deg2rad(P.angle_deg))
                )
                gt = project(truth, masks)
                expectation = forward_model(
                    emission, Acquisition(scene.axes, shape, output="expectation"), P
                )
                raw = digitize(expectation, P.camera, seed=CONFIG["camera_seed"])
                inputs = ROOT / "inputs"
                inputs.mkdir(exist_ok=True)
                np.savez_compressed(
                    inputs / f"{profile}_{kind}_{photons}.npz",
                    raw=raw,
                    expectation=expectation,
                )
                for factor, folder in zip(CONFIG["scan_factors"], folders, strict=True):
                    if not (folder / marker).exists():
                        generate_case(raw, gt, axes, masks, psf, factor, folder, kind)
                        print(
                            "RECONSTRUCTED", photons, profile, kind, factor, flush=True
                        )


def load(path: Path) -> dict[str, np.ndarray]:
    """Load final numerical panels without retaining an open archive."""
    with np.load(path) as data:
        return dict(data)


def condition(profile: str, factor: int) -> str:
    """Describe illumination and scan spacing from configuration."""
    return (
        "Uniform" if profile == "uniform" else "Edge-tapered"
    ) + f" illumination · {factor * P.scan_step_um:g} µm scan"


def line_figure(profile: str, factor: int, view: str) -> plt.Figure:
    """Plot physical line cuts for every configured photon budget."""
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(15, 6.4), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.38, top=0.80, wspace=0.25)
    fig.suptitle(
        "Paired-line intensity cuts · " + condition(profile, factor),
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.065,
        0.885,
        f"{view} maximum-intensity projection · camera pitch {P.pixel_size_um:g} µm · {P.angle_deg:g}° OPM",
        fontsize=12,
    )
    for ax, photons in zip(axes, LEVELS, strict=False):
        folder = ROOT / f"photons_{photons}/{profile}_line-pairs_f{factor}"
        data = load(folder / "profiles.npz")
        info = next(
            r
            for r in json.loads((folder / "profiles.json").read_text())
            if r["view"] == view
        )
        x = data[view + "_x"]
        for method in METHODS:
            iteration = {
                "rl_best": info["best_iteration"],
                "rl_end": RL_ITERATIONS,
                "gc": info["gc_iteration"],
            }.get(method)
            label = NAMES[method]
            if method != "gt":
                score = info["scores"][method]
                label += f" · I={iteration} · PSNR {score:.2f} dB"
                LABELS.append(
                    {
                        "figure": f"line_{profile}_f{factor}_{view}",
                        "photons": photons,
                        "view": view,
                        "method": method,
                        "iteration": iteration,
                        "psnr_db": score,
                    }
                )
            if method == "gt":
                ax.step(
                    x,
                    data[view + "_" + method],
                    where="mid",
                    color=COLORS[method],
                    linewidth=1.5,
                    label=label,
                )
            else:
                ax.plot(
                    x,
                    data[view + "_" + method],
                    color=COLORS[method],
                    linewidth=1.25,
                    linestyle="--" if method == "rl_end" else "-",
                    label=label,
                    marker="." if method == "rl_end" else None,
                    markersize=2,
                    markevery=3,
                )
        ax.set(
            title=f"{photons:,} foreground / {BACKGROUND[photons]} background photons",
            xlabel="X (µm)",
            xlim=(x[0], x[-1]),
            ylim=(0, None),
        )
        ax.set_ylabel("Intensity (calibrated count units)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.15)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.19),
            fontsize=8.5,
            frameon=False,
            borderaxespad=0,
        )
        if info["best_iteration"] == RL_ITERATIONS:
            ax.text(
                0.98,
                0.94,
                "Best recorded RL = endpoint",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
            )
    cut = json.loads(
        (
            ROOT / f"photons_{LEVELS[0]}/{profile}_line-pairs_f{factor}/profiles.json"
        ).read_text()
    )
    cut = next(r for r in cut if r["view"] == view)["cut_coordinate_um"]
    fig.text(
        0.065,
        0.04,
        f"Fixed cut at {'Y' if view == 'XY' else 'Z'} = {cut:.3f} µm. PSNR uses the full displayed GT projection, not the 1D cut.\n"
        f"Best recorded RL: highest PSNR among the saved iteration checkpoints through I={RL_ITERATIONS:,}. RLGC: MATLAB read-noise correction ({READ_NOISE:g} RMS), KL enabled.\n"
        f"Incident photon levels are before QE. Intensities are unscaled. Pair spacings: {GAPS} µm.",
        fontsize=9,
        linespacing=1.45,
    )
    return fig


def image_panel(
    ax: plt.Axes,
    image: np.ndarray,
    reference: np.ndarray,
    extent: tuple[float, ...],
    method: str,
    iteration: int | None,
    tag: dict,
) -> object:
    """Display an unscaled reconstruction and its GT-referenced PSNR."""
    maximum = float(reference.max())
    artist = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=maximum,
        interpolation="none",
        aspect="equal",
    )
    ax.set_axis_off()
    ax.text(
        0.025,
        0.975,
        NAMES[method],
        color="#169dd2" if method == "gc" else "white",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        transform=ax.transAxes,
        path_effects=STROKE,
    )
    if method != "gt":
        score = peak_signal_noise_ratio(
            reference, image, data_range=float(np.max(reference))
        )
        ax.text(
            0.975,
            0.90,
            f"PSNR {score:.2f} dB",
            color="white",
            ha="right",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
            path_effects=STROKE,
        )
        ax.text(
            0.025,
            0.035,
            f"I = {iteration}" + (" · automatic stop" if method == "gc" else ""),
            color="white",
            ha="left",
            va="bottom",
            fontsize=9,
            transform=ax.transAxes,
            path_effects=STROKE,
        )
        LABELS.append(dict(**tag, method=method, iteration=iteration, psnr_db=score))
    horizontal = extent[1] - extent[0]
    vertical = extent[3] - extent[2]
    scale = 5 if horizontal > 20 else 2
    right = extent[1] - 0.06 * horizontal
    bottom = extent[2] + 0.09 * vertical
    ax.plot(
        [right - scale, right],
        [bottom, bottom],
        color="white",
        linewidth=3,
        solid_capstyle="butt",
    )
    ax.text(
        right - scale / 2,
        bottom + 0.035 * vertical,
        f"{scale} µm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=9,
        path_effects=STROKE,
    )
    return artist


def physical_extent(data: dict[str, np.ndarray], view: str) -> tuple[float, ...]:
    """Return image bounds from recorded camera-grid coordinates."""
    vertical, horizontal = {"XY": ("y", "x"), "XZ": ("z", "x"), "YZ": ("z", "y")}[view]
    v, h = data[vertical], data[horizontal]
    p = P.pixel_size_um
    return h[0] - p / 2, h[-1] + p / 2, v[0] - p / 2, v[-1] + p / 2


def sphere_figure(photons: int, profile: str, factor: int) -> plt.Figure:
    """Compare the same sphere volumes in three orthogonal views."""
    folder = ROOT / f"photons_{photons}/{profile}_sphere_f{factor}"
    data = load(folder / "views.npz")
    info = json.loads((folder / "views.json").read_text())
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 11.2))
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.095, top=0.89, wspace=0.035, hspace=0.10
    )
    fig.suptitle(
        "Hollow sphere · " + condition(profile, factor),
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.065,
        0.93,
        f"{photons:,} foreground / {BACKGROUND[photons]} background photons per camera pixel before QE · {P.pixel_size_um:g} µm pixels",
        fontsize=12,
    )
    for row, view in enumerate(("XY", "XZ", "YZ")):
        for column, method in enumerate(METHODS):
            iteration = {
                "rl_best": info["best_iteration"],
                "rl_end": RL_ITERATIONS,
                "gc": info["gc_iteration"],
            }.get(method)
            image_panel(
                axes[row, column],
                data[method + "_" + view],
                data["gt_" + view],
                physical_extent(data, view),
                method,
                iteration,
                {
                    "figure": f"sphere_{profile}_f{factor}_{photons}",
                    "photons": photons,
                    "view": view,
                },
            )
        axes[row, 0].text(
            -0.07,
            0.5,
            view,
            rotation=90,
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
            transform=axes[row, 0].transAxes,
        )
    fig.text(
        0.065,
        0.025,
        "XY, XZ and YZ maximum-intensity projections. Each row shares its GT intensity scale; brighter values saturate white.\n"
        f"RL best recorded: highest saved XZ PSNR at I={info['best_iteration']}; the same 3D reconstruction is shown in all views.\n"
        f"RL endpoint I={RL_ITERATIONS:,}. RLGC uses MATLAB read-noise correction ({READ_NOISE:g} RMS) and reference stopping with KL enabled. No intensity fitting.",
        fontsize=9,
        linespacing=1.4,
    )
    return fig


def star_figure(profile: str, factor: int) -> plt.Figure:
    """Compare the configured Siemens target across photon budgets."""
    fig, axes = plt.subplots(len(LEVELS), 4, figsize=(12.5, 11.4), squeeze=False)
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.105, top=0.89, wspace=0.035, hspace=0.10
    )
    fig.suptitle(
        "MATLAB Siemens target · " + condition(profile, factor),
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.065,
        0.935,
        f"{STAR['cycles']} sinusoidal spokes · half-bright center · {STAR_SAMPLES}×{STAR_SAMPLES} fine GT samples · {P.pixel_size_um:g} µm camera pixels",
        fontsize=12,
    )
    for row, photons in enumerate(LEVELS):
        folder = ROOT / f"photons_{photons}/{profile}_siemens-star_f{factor}"
        data = load(folder / "views.npz")
        info = json.loads((folder / "views.json").read_text())
        for column, method in enumerate(METHODS):
            iteration = {
                "rl_best": info["best_iteration"],
                "rl_end": RL_ITERATIONS,
                "gc": info["gc_iteration"],
            }.get(method)
            image_panel(
                axes[row, column],
                data[method + "_XY"],
                data["gt_XY"],
                physical_extent(data, "XY"),
                method,
                iteration,
                {
                    "figure": f"star_{profile}_f{factor}",
                    "photons": photons,
                    "view": "XY",
                },
            )
        axes[row, 0].text(
            -0.07,
            0.5,
            f"{photons:,}/{BACKGROUND[photons]} photons",
            rotation=90,
            ha="right",
            va="center",
            fontsize=11,
            transform=axes[row, 0].transAxes,
        )
    fig.text(
        0.065,
        0.025,
        "XY maximum-intensity projections. Rows: source/background incident photons per camera pixel, before QE. Shared GT scale within each row.\n"
        f"MATLAB angular target in a {STAR['thickness_um'] * 1000:g} nm fluorescent XY sheet, propagated through the shared {P.angle_deg:g}° OPM fluorescence/camera model.\n"
        f"Fine GT pitch {P.fine_spacing_um:g} µm; camera pitch {P.pixel_size_um:g} µm. RL best: highest recorded XY PSNR through I={RL_ITERATIONS:,}. RLGC: {READ_NOISE:g} RMS, KL enabled.\n"
        "The MATLAB brightfield OTF is not used as an OPM forward model. No reconstruction intensity fitting.",
        fontsize=9,
        linespacing=1.4,
    )
    return fig


def save(fig: plt.Figure, stem: str, pdf: PdfPages) -> None:
    """Export a labeled figure and append it to its combined PDF."""
    fig.savefig(OUT / f"{stem}.png", dpi=CONFIG["rendering"]["dpi"])
    fig.savefig(OUT / f"{stem}.pdf")
    pdf.savefig(fig)
    plt.close(fig)
    print("FIGURE", stem, flush=True)


def points_figure(profile: str, factor: int) -> plt.Figure:
    """Compare point reconstructions in the laboratory XZ view."""
    fig, axes = plt.subplots(len(LEVELS), 4, figsize=(12.5, 11.4), squeeze=False)
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.105, top=0.89, wspace=0.035, hspace=0.10
    )
    fig.suptitle(
        "Points · " + condition(profile, factor), fontsize=16, fontweight="bold", y=0.98
    )
    fig.text(
        0.065,
        0.935,
        f"XZ maximum-intensity projections · {P.pixel_size_um:g} µm camera pixels · {P.angle_deg:g}° OPM",
        fontsize=12,
    )
    for row, photons in enumerate(LEVELS):
        folder = ROOT / f"photons_{photons}/{profile}_points_f{factor}"
        data = load(folder / "views.npz")
        info = json.loads((folder / "views.json").read_text())
        for column, method in enumerate(METHODS):
            iteration = {
                "rl_best": info["best_iteration"],
                "rl_end": RL_ITERATIONS,
                "gc": info["gc_iteration"],
            }.get(method)
            image_panel(
                axes[row, column],
                data[method + "_XZ"],
                data["gt_XZ"],
                physical_extent(data, "XZ"),
                method,
                iteration,
                {
                    "figure": f"points_{profile}_f{factor}",
                    "photons": photons,
                    "view": "XZ",
                },
            )
        axes[row, 0].text(
            -0.07,
            0.5,
            f"{photons:,}/{BACKGROUND[photons]} photons",
            rotation=90,
            ha="right",
            va="center",
            fontsize=11,
            transform=axes[row, 0].transAxes,
        )
    fig.text(
        0.065,
        0.025,
        "Rows: source/background incident photons per camera pixel before QE. Shared GT display scale within each row.\n"
        "Zero-background conditions retain source shot noise and camera read noise.\n"
        f"RL best: highest recorded XZ PSNR through I={RL_ITERATIONS:,}. RLGC: reference read-noise correction ({READ_NOISE:g} RMS), KL enabled.\n"
        "No reconstruction intensity fitting.",
        fontsize=9,
        linespacing=1.4,
    )
    return fig


def main() -> None:
    """Generate or render the explicitly configured comparison series."""
    global \
        CONFIG, \
        ROOT, \
        OUT, \
        P, \
        LEVELS, \
        BACKGROUND, \
        CONDITIONS, \
        RL_ITERATIONS, \
        READ_NOISE, \
        STAR, \
        STAR_SAMPLES, \
        GAPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    CONFIG = json.loads(args.config.read_text())
    ROOT = args.output
    ROOT.mkdir(parents=True, exist_ok=True)
    configuration_path = ROOT / "configuration.json"
    if configuration_path.exists():
        saved = json.loads(configuration_path.read_text())
        if {k: v for k, v in saved.items() if k != "rendering"} != {
            k: v for k, v in CONFIG.items() if k != "rendering"
        }:
            raise ValueError(
                "The output contains a different physical or reconstruction configuration; choose a new output directory"
            )
    OUT = ROOT / "figures"
    OUT.mkdir(exist_ok=True)
    settings = dict(CONFIG["microscope"])
    settings["camera"] = Camera(**settings["camera"])
    settings["psf_half_extent_zyx_um"] = tuple(settings["psf_half_extent_zyx_um"])
    P = Microscope(**settings)
    LEVELS = tuple(row[0] for row in CONFIG["photon_budgets"])
    BACKGROUND = dict(CONFIG["photon_budgets"])
    CONDITIONS = [
        (profile, factor)
        for profile in CONFIG["illumination_profiles"]
        for factor in CONFIG["scan_factors"]
    ]
    RL_ITERATIONS = CONFIG["rl"]["iterations"]
    READ_NOISE = CONFIG["rlgc"]["read_noise"]
    if RL_ITERATIONS not in CONFIG["rl"]["checkpoints"]:
        raise ValueError("RL endpoint must be included in the recorded checkpoints")
    STAR = CONFIG["specimens"]["siemens-star"]["features"]
    STAR_SAMPLES = round(2 * STAR["radius_um"] / P.fine_spacing_um) + 1
    GAPS = ", ".join(map(str, CONFIG["specimens"]["line-pairs"]["features"]["gaps_um"]))
    plt.rcParams.update(
        {
            "font.family": CONFIG["rendering"]["font_family"],
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    if not args.render_only:
        simulate()
    LABELS.clear()
    for name, kind in (
        ("line_cuts", "line-pairs"),
        ("sphere_orthogonal", "sphere"),
        ("siemens_star", "siemens-star"),
        ("points", "points"),
    ):
        if kind not in CONFIG["kinds"]:
            continue
        with PdfPages(OUT / f"{name}.pdf") as pdf:
            for profile, factor in CONDITIONS:
                if kind == "line-pairs":
                    for view in ("XY", "XZ"):
                        save(
                            line_figure(profile, factor, view),
                            f"line_{profile}_f{factor}_{view}",
                            pdf,
                        )
                elif kind == "sphere":
                    for photons in LEVELS:
                        save(
                            sphere_figure(photons, profile, factor),
                            f"sphere_{profile}_f{factor}_{photons}",
                            pdf,
                        )
                elif kind == "siemens-star":
                    save(star_figure(profile, factor), f"star_{profile}_f{factor}", pdf)
                else:
                    save(
                        points_figure(profile, factor),
                        f"points_{profile}_f{factor}",
                        pdf,
                    )
    with (OUT / "psnr_labels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(LABELS[0]))
        writer.writeheader()
        writer.writerows(LABELS)
    (ROOT / "configuration.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    (OUT / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Deconvolution figures</title>'
        "<h1>Deconvolution comparisons</h1>"
        + "".join(
            f'<p><a href="{name}.pdf">{label}</a></p>'
            for name, label in [
                ("line_cuts", "Line cuts"),
                ("sphere_orthogonal", "Sphere XY/XZ/YZ"),
                ("siemens_star", "Siemens star"),
                ("points", "Points"),
            ]
        )
    )
    print("COMPLETE", len(LABELS), "PSNR labels", flush=True)


if __name__ == "__main__":
    main()
