"""House plot style for figures made for the slides (LaTeX-style serif text, 18 pt labels,
15 pt ticks, outward ticks, 2 pt lines), plus a save helper that refuses to write a figure
with overlapping or clipped text."""
import shutil

import matplotlib.pyplot as plt


def apply():
    # fontstyle
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    # fontsizes
    plt.rc("font", size=18)
    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    plt.rc("legend", fontsize=18)

    # tick format
    plt.rc("xtick", top=False)
    plt.rc("ytick", right=False)
    plt.rc("xtick", direction="out")
    plt.rc("ytick", direction="out")
    plt.rc("xtick.major", size=5.0)
    plt.rc("ytick.major", size=5.0)
    plt.rc("xtick.major", width=1.0)
    plt.rc("ytick.major", width=1.0)

    # line format
    plt.rc("lines", linewidth=2)
    plt.rc("lines", markersize=6)
    plt.rc("lines", markeredgewidth=1)

    # Without a LaTeX install (latex + dvipng), fall back to matplotlib's Computer Modern.
    if not (shutil.which("latex") and shutil.which("dvipng")):
        plt.rc("text", usetex=False)
        plt.rc("mathtext", fontset="cm")
        plt.rc("font", serif=["cmr10", "DejaVu Serif"])
        plt.rc("axes.formatter", use_mathtext=True)


def clean_axes(ax, grid_axis="y"):
    """No top/right spines; a light grid on the value axis only, behind the data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#e1e0d9", lw=0.8)


def save_checked(fig, path, dpi=200, pad=0.0):
    """Save `fig`, but raise instead if any two visible texts overlap or a text is cut off."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = []
    for t in fig.findobj(match=plt.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer=r)
        if bb.width == 0 or bb.height == 0:
            continue
        texts.append((t.get_text(), bb))
    problems = []
    for name, bb in texts:
        if bb.x0 < -pad or bb.y0 < -pad or bb.x1 > W + pad or bb.y1 > H + pad:
            problems.append(f"cut off: {name!r}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (a, ba), (b, bb) = texts[i], texts[j]
            if ba.overlaps(bb):
                problems.append(f"overlap: {a!r} / {b!r}")
    if problems:
        raise RuntimeError(f"{path}: " + "; ".join(problems))
    fig.savefig(path, dpi=dpi)
