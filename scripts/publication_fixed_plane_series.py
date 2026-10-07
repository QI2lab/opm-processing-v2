"""Generate fixed-objective frame figures for four sphere scan spacings.

Read saved 0.4, 0.8, 1.2, and 1.6 micrometer acquisitions and their deconvolution
outputs, plus the fine Cartesian truth and PSF from the acquisition parent.
Independently convolve the object and sample a fixed detection plane to compare
acquired frames, reconstructions on a 0.4 micrometer scan grid, and missing-plane
patterns. Write PNG, SVG, and PDF figures, per-spacing NPZ source arrays, and
manifest.json provenance to --output. Requires matplotlib; run as
``uv run --with matplotlib python -m scripts.publication_fixed_plane_series --acquisitions DIR1 DIR2 DIR3 DIR4 --output DIR``
from the repository root.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve
from tifffile import imread


def physical_axes(metadata):
    """Expand recorded coordinates to physical voxel centers.

    Parameters
    ----------
    metadata
        Saved axis records containing coordinate origin, step, and sample count.

    Returns
    -------
    tuple[np.ndarray, ...]
        Physical coordinate arrays reconstructed from the saved axis records.
    """
    return tuple(a["origin_um"] + np.arange(a["size"]) * a["step_um"] for a in metadata)


def fixed_plane_frames(field, axes, scans, rows, columns, theta, pitch, samples=1):
    """Translate a Cartesian field past a fixed detector, optionally averaging pixels.

    Parameters
    ----------
    field
        Cartesian fluorescence or optical field sampled by the experiment.
    axes
        Physical coordinate arrays for the source image, in storage-axis order.
    scans
        Selected scan-plane positions in micrometers.
    rows
        Detector-row center coordinates in micrometers.
    columns
        Detector-column center coordinates in micrometers.
    theta
        Detector plane angle in radians.
    pitch
        Detector pixel pitch in micrometers.
    samples
        Quadrature samples per detector pixel along each camera axis.

    Returns
    -------
    np.ndarray
        Pixel-integrated detector frames for the selected fixed-plane translations.
    """
    frames = []
    offsets = ((np.arange(samples) + 0.5) / samples - 0.5) * pitch
    for scan in scans:
        interpolate = RegularGridInterpolator(
            (axes[0], axes[1] - scan, axes[2]), field, bounds_error=False, fill_value=0
        )
        frame = np.zeros((len(rows), len(columns)), dtype=np.float32)
        for dr in offsets:
            for dx in offsets:
                rr, xx = np.meshgrid(rows + dr, columns + dx, indexing="ij")
                frame += interpolate(
                    np.stack((rr * np.sin(theta), rr * np.cos(theta), xx), axis=-1)
                )
        frames.append(frame / samples**2)
    return np.asarray(frames)


def diagram(ax, scan, theta, radius=5):
    """Show a stationary plane while the object translates in laboratory Y.

    Parameters
    ----------
    ax
        Matplotlib axes receiving the drawing.
    scan
        Object translation along the acquisition scan axis, in micrometers.
    theta
        Detector plane angle in radians.
    radius
        Sphere radius displayed in the geometry sketch, in micrometers.
    """
    displacement = 0.0 if abs(scan) < 1e-8 else -scan
    ax.add_patch(Circle((displacement, 0), radius, facecolor="0.9", edgecolor="0.35"))
    rr = np.array([-25, 25])
    ax.plot(rr * np.cos(theta), rr * np.sin(theta), color="#0072B2", lw=1.5)
    ax.plot(displacement, 0, "+", color="0.35")
    ax.set(
        title=f"Object Y = {displacement:.1f} µm",
        xlim=(-18, 18),
        ylim=(-9, 9),
        xlabel="Laboratory Y (µm)",
        ylabel="Z (µm)",
        aspect="equal",
    )


def camera_panel(ax, frame, axes, title, vmax):
    """Render one frame with physical coordinates and unobstructed image pixels.

    Parameters
    ----------
    ax
        Matplotlib axes receiving the drawing.
    frame
        Camera frame record containing event indices and instrument metadata.
    axes
        Physical coordinate arrays for the source image, in storage-axis order.
    title
        Title displayed above the camera panel.
    vmax
        Upper image display intensity; the lower limit is zero.

    Returns
    -------
    matplotlib.image.AxesImage
        Displayed camera image artist with physical row and column coordinates.
    """
    rows, columns = axes
    pitch = columns[1] - columns[0]
    extent = (
        columns[0] - pitch / 2,
        columns[-1] + pitch / 2,
        rows[0] - pitch / 2,
        rows[-1] + pitch / 2,
    )
    im = ax.imshow(
        frame,
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set(
        xlim=(-8, 8),
        ylim=(-16, 16),
        xlabel="Column (µm)",
        ylabel="Row (µm)",
        title=title,
    )
    ax.set_facecolor("black")
    return im


def save(fig, directory, stem):
    """Export the same layout with vector labels and publication-resolution pixels.

    Parameters
    ----------
    fig
        Matplotlib figure to write and close.
    directory
        Output directory for the figure files.
    stem
        Output filename stem without the figure extension.
    """
    for extension in ("pdf", "svg", "png"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=600)
    plt.close(fig)


def main():
    """Prepare matching acquisition, reconstruction and missing-plane comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisitions", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    metas = [json.loads((d / "simulation.json").read_text()) for d in args.acquisitions]
    p = metas[0]["parameters"]
    theta = np.deg2rad(p["angle_deg"])
    pitch = p["pixel_size_um"]
    truth = imread(args.acquisitions[0].parent / "ground_truth_fine.tif")
    psf = imread(args.acquisitions[0].parent / "psf_cartesian_fine.tif")
    print(
        "Independent Cartesian optical convolution for fixed-plane predictions",
        flush=True,
    )
    blurred = np.maximum(fftconvolve(truth, psf, mode="same"), 0)
    fine_axes = physical_axes(metas[0]["fine_cartesian_axes_zyx"])
    common_scans = np.array([-10.4, -8, -4, 0, 4, 8, 10.4], dtype=float)
    consecutive_scans = np.arange(-4, 5) * 0.4
    manifest = dict(
        parameters=p,
        reconstruction_step_um=0.4,
        cases=[],
        reconstruction_display_scan_positions_um=common_scans.tolist(),
        consecutive_scan_positions_um=consecutive_scans.tolist(),
        camera_display_limits=[0, 0.5],
        reconstruction_display_limits=[0, 1],
    )
    overviews = []
    for factor, (directory, meta) in enumerate(zip(args.acquisitions, metas), 1):
        step = 0.4 * factor
        assert meta["parameters"] == p
        assert np.isclose(meta["acquisition"]["scan_step_um"], step)
        for key in (
            "camera_samples",
            "peak_electrons",
            "background_electrons",
            "read_noise_e",
            "seed",
        ):
            assert meta["acquisition"][key] == metas[0]["acquisition"][key]
        scans, rows, columns = physical_axes(meta["raw_axes_syx"])
        raw = imread(directory / "raw_skewed.tif")
        indices = np.asarray([np.argmin(abs(scans - s)) for s in common_scans])
        selected_scans = scans[indices]
        predictions = fixed_plane_frames(
            blurred,
            fine_axes,
            selected_scans,
            rows,
            columns,
            theta,
            pitch,
            meta["acquisition"]["camera_samples"],
        )
        print(f"Rendering {step:.1f} µm acquisition and reconstruction", flush=True)
        fig, axs = plt.subplots(
            3, len(common_scans), figsize=(16, 9), layout="constrained"
        )
        for col, s in enumerate(selected_scans):
            diagram(axs[0, col], s, theta)
            camera_panel(
                axs[1, col],
                predictions[col],
                (rows, columns),
                "Fixed-plane prediction",
                0.5,
            )
            im = camera_panel(
                axs[2, col],
                raw[indices[col]],
                (rows, columns),
                "Measured noisy frame",
                0.5,
            )
        fig.colorbar(im, ax=axs[1:, :], shrink=0.5, label="Fluorescence density (a.u.)")
        fig.suptitle(
            f"{step:.1f} µm acquisition: fixed oblique plane, translating object\n"
            "Seven acquired frames across the sphere; actual positions labeled",
            fontsize=12,
        )
        save(fig, args.output, f"acquisition_{step:.1f}um")
        decon_dir = directory / (
            "decon_native" if factor == 1 else f"decon_upsample{factor}"
        )
        dm = json.loads((decon_dir / "simulation.json").read_text())
        assert np.isclose(dm["deconvolution"]["reconstruction_step_um"], 0.4)
        raw_hash = hashlib.sha256(
            (directory / "raw_electrons.tif").read_bytes()
        ).hexdigest()
        assert raw_hash == dm["deconvolution"]["raw_file_sha256"]
        rec = imread(decon_dir / "deconvolved_skewed.tif")
        assert np.isfinite(rec).all()
        rec_indices = np.rint((common_scans - scans[0]) / 0.4).astype(int)
        np.testing.assert_allclose(
            scans[0] + rec_indices * 0.4, common_scans, atol=1e-9
        )
        assert rec.shape[0] == (len(scans) - 1) * factor + 1
        assert rec.shape[1:] == raw.shape[1:]
        reference = imread(decon_dir / "normal.ome.tif")
        deskewed = imread(decon_dir / "deskewed.ome.tif")
        cart_axes = physical_axes(dm["output_axes_zyx"])
        reference_frames = fixed_plane_frames(
            reference, cart_axes, common_scans, rows, columns, theta, pitch
        )
        deskewed_frames = fixed_plane_frames(
            deskewed, cart_axes, common_scans, rows, columns, theta, pitch
        )
        acquired = np.array([np.min(abs(scans - s)) < 1e-8 for s in common_scans])
        fig, axs = plt.subplots(
            4, len(common_scans), figsize=(16, 12), layout="constrained"
        )
        for col, s in enumerate(common_scans):
            diagram(axs[0, col], s, theta)
            camera_panel(
                axs[1, col],
                reference_frames[col],
                (rows, columns),
                "Cartesian GC reference",
                1,
            )
            status = "acquired position" if acquired[col] else "unmeasured position"
            camera_panel(
                axs[2, col],
                rec[rec_indices[col]],
                (rows, columns),
                f"OPM GC: {status}",
                1,
            )
            im = camera_panel(
                axs[3, col],
                deskewed_frames[col],
                (rows, columns),
                "After orthogonal deskew*",
                1,
            )
        fig.colorbar(im, ax=axs[1:, :], shrink=0.5, label="Fluorescence density (a.u.)")
        fig.suptitle(
            f"{step:.1f} → 0.4 µm reconstruction: same fixed-plane positions\n"
            "*Cartesian result re-sliced onto the oblique plane for frame comparison",
            fontsize=12,
        )
        save(fig, args.output, f"reconstruction_{step:.1f}_to_0.4um")
        consecutive_raw = np.zeros(
            (len(consecutive_scans), *raw.shape[1:]), dtype=np.float32
        )
        measured = np.zeros(len(consecutive_scans), dtype=bool)
        for j, s in enumerate(consecutive_scans):
            ix = int(np.argmin(abs(scans - s)))
            if abs(scans[ix] - s) < 1e-8:
                consecutive_raw[j] = raw[ix]
                measured[j] = True
        ix = np.rint((consecutive_scans - scans[0]) / 0.4).astype(int)
        np.testing.assert_allclose(scans[0] + ix * 0.4, consecutive_scans, atol=1e-9)
        consecutive_ref = fixed_plane_frames(
            reference, cart_axes, consecutive_scans, rows, columns, theta, pitch
        )
        overviews.append(
            dict(
                step=step,
                raw=consecutive_raw,
                measured=measured,
                rec=rec[ix],
                reference=consecutive_ref,
                axes=(rows, columns),
            )
        )
        np.savez_compressed(
            args.output / f"frame_source_{step:.1f}um.npz",
            acquired_scan_positions_um=selected_scans,
            fixed_plane_predictions=predictions,
            acquired_frames=raw[indices],
            reconstruction_scan_positions_um=common_scans,
            reference_frames=reference_frames,
            reconstructed_frames=rec[rec_indices],
            deskewed_resliced_frames=deskewed_frames,
            acquired_position_mask=acquired,
            camera_rows_um=rows,
            camera_columns_um=columns,
            consecutive_scan_positions_um=consecutive_scans,
            consecutive_raw=consecutive_raw,
            consecutive_acquired_mask=measured,
            consecutive_reconstruction=rec[ix],
            consecutive_reference=consecutive_ref,
        )
        manifest["cases"].append(
            dict(
                acquisition=str(directory.resolve()),
                reconstruction=str(decon_dir.resolve()),
                scan_step_um=step,
                acquired_planes=len(scans),
                reconstructed_planes=rec.shape[0],
                acquired_scan_positions_um=selected_scans.tolist(),
                raw_electrons_sha256=raw_hash,
                metrics=dm["metrics"],
            )
        )
    for stage in ("raw", "rec"):
        fig, axs = plt.subplots(5, 9, figsize=(18, 12), layout="constrained")
        for col, s in enumerate(consecutive_scans):
            diagram(axs[0, col], s, theta)
        for row, case in enumerate(overviews, 1):
            for col, s in enumerate(consecutive_scans):
                missing = not case["measured"][col]
                status = "measured" if not missing else "inferred"
                im = camera_panel(
                    axs[row, col],
                    case[stage][col],
                    case["axes"],
                    f"{case['step']:.1f} µm"
                    if stage == "raw"
                    else f"{case['step']:.1f} µm: {status}",
                    0.5 if stage == "raw" else 1,
                )
        fig.colorbar(im, ax=axs[1:, :], shrink=0.5, label="Fluorescence density (a.u.)")
        fig.suptitle(
            (
                "Acquisition: unmeasured positions are shown as zeros"
                if stage == "raw"
                else "GC reconstruction: measured and inferred scan positions"
            )
            + "\nConsecutive 0.4 µm scan positions; fixed oblique plane, translating sphere",
            fontsize=13,
        )
        save(
            fig,
            args.output,
            "consecutive_acquisitions"
            if stage == "raw"
            else "consecutive_reconstructions",
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Figures and source data written to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
