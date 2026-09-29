"""Explain the unchanged production orthogonal interpolation with vector diagrams."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Polygon
import numpy as np

from opm_processing.imageprocessing.opmtools import orthogonal_deskew


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#00856A"


def arrow(ax, start, end, color="black", style="-|>", **kwargs):
    """Draw a vector arrow in diagram coordinates."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=1.5, **kwargs),
    )


def project(x, y, z):
    """Use an explicit axonometric view preserving the YZ geometry."""
    return np.array([y - 0.75 * x, z + 0.30 * x])


def main():
    """Render plane geometry, the four-sample stencil, and exact implementation rules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    theta = np.deg2rad(30)
    tangent = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])
    spacing, pitch = 3.0, 1.0  # Schematic dimensions, not acquisition settings.
    point = np.array([5.0, 2.0])  # (Y,Z), one exact output grid point.
    virtual_scan = point[0] - point[1] / np.tan(theta)
    feet, fractions, lower = [], [], []
    for si in (0.0, spacing):
        vi = np.dot(point - [si, 0], tangent)
        qi = np.array([si, 0]) + vi * tangent
        np.testing.assert_allclose(np.dot(point - qi, tangent), 0, atol=1e-12)
        np.testing.assert_allclose(qi[0] - qi[1] / np.tan(theta), si, atol=1e-12)
        feet.append(qi)
        lower.append(int(np.floor(vi / pitch)))
        fractions.append(vi / pitch - lower[-1])
    assert 0 < virtual_scan < spacing
    # Check the shown stencil against the real implementation, including the
    # existing scale factor and unchanged camera-column index.
    data = np.zeros((4, 16, 3), dtype=np.float32)
    neighbor_values = ((10.0, 30.0), (20.0, 50.0))
    for plane, (k, values) in enumerate(zip(lower, neighbor_values)):
        data[plane, k : k + 2, 1] = values
    interpolated = [
        (1 - w) * a + w * b for w, (a, b) in zip(fractions, neighbor_values)
    ]
    expected = pitch / spacing * sum(interpolated)
    result = orthogonal_deskew(
        data,
        theta=30,
        distance=spacing,
        pixel_size=pitch,
        downsample_factor=1,
        divisible_by=1,
    )
    np.testing.assert_allclose(result[2, 5, 1], expected, rtol=1e-6)
    np.testing.assert_array_equal(result[:, :, [0, 2]], 0)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(12.8, 6.0), facecolor="white")
    gs = fig.add_gridspec(
        1,
        2,
        left=0.04,
        right=0.98,
        bottom=0.20,
        top=0.91,
        wspace=0.10,
        hspace=0.12,
    )
    ax = fig.add_subplot(gs[0, 0])
    for si, color, label in (
        (spacing, ORANGE, r"$\Pi_{i+1}$"),
        (0.0, BLUE, r"$\Pi_i$"),
    ):

        def location(u, v):
            return project(u, si + v * np.cos(theta), v * np.sin(theta))

        corners = [location(u, v) for u, v in ((0, 0), (3.4, 0), (3.4, 7.2), (0, 7.2))]
        ax.add_patch(
            Polygon(corners, facecolor=color, alpha=0.17, edgecolor=color, lw=1.8)
        )
        for u in np.linspace(0, 3.4, 6):
            line = np.asarray([location(u, v) for v in (0, 7.2)])
            ax.plot(*line.T, color=color, lw=0.55, alpha=0.55)
        for v in np.arange(0, 7.3, 0.9):
            line = np.asarray([location(u, v) for u in (0, 3.4)])
            ax.plot(*line.T, color=color, lw=0.55, alpha=0.55)
        edge = np.asarray([location(0, 0), location(0, 7.2)])
        ax.plot(*edge.T, color=color, lw=2)
        top = location(0, 7.2)
        ax.text(top[0] - 0.8, top[1] + 0.25, label, color=color, fontsize=15)
    origin = project(0, 0, 0)
    arrow(ax, origin, project(4.2, 0, 0))
    ax.text(-3.0, 1.7, "$X=u$\n(µm)", ha="center", fontsize=12)
    arrow(ax, origin, project(0, 0, 4.8))
    ax.text(-0.2, 4.8, "$Z$ (µm)", ha="right", fontsize=12)
    arrow(ax, (0, -1.0), (9.0, -1.0))
    ax.text(
        7.0, -1.45, "Lab $Y$ (µm)\nScan displacement $s$ (µm)", ha="center", va="top", fontsize=10
    )
    arrow(ax, (0, -0.45), (spacing, -0.45), style="<->")
    ax.text(spacing / 2, -0.25, r"$\Delta s$", ha="center", fontsize=12)
    ax.text(0, -1.5, r"$s_i$", ha="center")
    ax.text(spacing, -1.5, r"$s_{i+1}$", ha="center")
    arrow(ax, 0.8 * tangent, 4.8 * tangent, BLUE)
    ax.text(2.7, 1.8, "$v$ (µm)", rotation=30, color=BLUE, fontsize=11)
    ax.add_patch(Arc((0, 0), 2.2, 2.2, theta1=0, theta2=30, lw=1))
    ax.text(1.18, 0.35, r"$\theta$")
    ax.set(xlim=(-4.0, 10), ylim=(-2.8, 5.4), aspect="equal")
    ax.set_anchor("N")
    ax.axis("off")
    ax.set_title("A", loc="left", fontsize=16, weight="bold", pad=12)
    ax = fig.add_subplot(gs[0, 1])
    lab_y, lab_z = np.meshgrid(np.arange(2, 8.1), np.arange(0, 4.5))
    other_points = (lab_y != point[0]) | (lab_z != point[1])
    ax.scatter(
        lab_y[other_points],
        lab_z[other_points],
        s=16,
        marker="o",
        color="0.77",
        edgecolors="none",
        zorder=0,
    )
    selected_pairs = [
        np.array([si, 0]) + np.arange(k, k + 2)[:, None] * pitch * tangent
        for si, k in zip((0, spacing), lower)
    ]
    # Outline the actual four contributing samples. The connector edges need
    # not be perpendicular: orthogonality refers to P's projections onto planes.
    corners = np.array(
        [
            selected_pairs[0][0],
            selected_pairs[0][1],
            selected_pairs[1][1],
            selected_pairs[1][0],
        ]
    )
    ax.add_patch(
        Polygon(
            corners,
            closed=True,
            facecolor=GREEN,
            alpha=0.10,
            edgecolor="none",
            zorder=1,
        )
    )
    outline = np.vstack((corners, corners[0]))
    ax.plot(*outline.T, color="0.40", lw=1.2, zorder=2)
    for pixel in corners:
        ax.plot(
            *np.array([pixel, point]).T, color="0.60", lw=0.9, linestyle=":", zorder=2
        )
    for j, (si, color) in enumerate(((0, BLUE), (spacing, ORANGE))):
        t = np.linspace(0, 9, 2)
        line = np.array([si, 0])[:, None] + tangent[:, None] * t
        ax.plot(*line, color=color, lw=2)
        v = np.arange(0, 10) * pitch
        points = np.array([si, 0])[:, None] + tangent[:, None] * v
        ax.scatter(*points, s=20, color=color, zorder=3)
        selected = points[:, lower[j] : lower[j] + 2]
        ax.plot(*selected, color=color, lw=3.0, zorder=4)
        ax.scatter(*selected, s=85, color=color, edgecolor="white", zorder=5)
        qi = feet[j]
        ax.scatter(*qi, s=65, facecolor="white", edgecolor=color, lw=1.7, zorder=6)
        toward = (point - qi) / np.linalg.norm(point - qi)
        square = np.array(
            [qi + 0.16 * tangent, qi + 0.16 * (tangent + toward), qi + 0.16 * toward]
        )
        ax.plot(*square.T, color=GREEN, lw=1.8, zorder=6)
        ax.annotate(
            "",
            xy=point,
            xytext=qi,
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.1),
            zorder=5,
        )
    ax.scatter(*point, marker="*", s=165, color="black", zorder=7)
    ax.text(5.35, 2.08, r"$P=(X,Y,Z)$", fontsize=12)
    ax.annotate(
        r"$Q_i$",
        xy=feet[0],
        xytext=(3.65, 3.3),
        color=BLUE,
        arrowprops=dict(arrowstyle="-", color=BLUE),
    )
    ax.annotate(
        r"$Q_{i+1}$",
        xy=feet[1],
        xytext=(6.15, 0.85),
        color=ORANGE,
        arrowprops=dict(arrowstyle="-", color=ORANGE),
    )
    for text, xy, at, color in (
        (r"$I_i[k_i]$", 5 * tangent, (3.0, 2.25), BLUE),
        (r"$I_i[k_i+1]$", 6 * tangent, (5.35, 3.4), BLUE),
        (
            r"$I_{i+1}[k_{i+1}]$",
            np.array([spacing, 0]) + 2 * tangent,
            (3.05, 0.55),
            ORANGE,
        ),
        (
            r"$I_{i+1}[k_{i+1}+1]$",
            np.array([spacing, 0]) + 3 * tangent,
            (6.15, 1.65),
            ORANGE,
        ),
    ):
        ax.annotate(
            text,
            xy=xy,
            xytext=at,
            color=color,
            fontsize=10,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )
    ax.text(2.2, 4.05, "$X=u$ (µm)", fontsize=11)
    ax.text(7.35, 2.2, r"$\Pi_{i+1}$", color=ORANGE, fontsize=13)
    ax.text(6.3, 4.0, r"$\Pi_i$", color=BLUE, fontsize=13)
    ax.set(
        xlim=(2, 8),
        ylim=(0, 4.4),
        xlabel="Lab $Y$ (µm)",
        ylabel="Lab $Z$ (µm)",
        aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("B", loc="left", fontsize=16, weight="bold", pad=12)
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", label=r"Plane $\Pi_i$ and pixels"),
            Line2D(
                [], [], color=ORANGE, marker="o", label=r"Plane $\Pi_{i+1}$ and pixels"
            ),
            Line2D([], [], color=GREEN, lw=2, label="Orthogonal projection"),
            Line2D(
                [],
                [],
                color="0.77",
                marker="o",
                linestyle="none",
                label="Other lab-grid points",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="*",
                markersize=11,
                linestyle="none",
                label=r"Target $P$",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(args.output / f"orthogonal_interpolation.{suffix}", dpi=600)
    plt.close(fig)
    report = dict(
        plane_indices=[0, 1],
        target_yz=point.tolist(),
        virtual_scan=virtual_scan,
        projected_points_yz=np.asarray(feet).tolist(),
        lower_row_indices=lower,
        row_fractions=fractions,
        interpolation_values=interpolated,
        expected_output=expected,
        production_output=float(result[2, 5, 1]),
        untouched_columns_zero=True,
        schematic_pitch=pitch,
        schematic_scan_step=spacing,
        plane_normal_yz=normal.tolist(),
        production_scale="p / delta_s times the sum of the two within-plane interpolants",
    )
    (args.output / "geometry_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
