"""Regenerates the README figures (no dataset images are used): python assets/make_figures.py

Text is auto-fitted: every label is measured after drawing and its font size is reduced
until it sits inside its box with padding, so nothing overflows or touches a neighbour.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent
INK, MUTED, BG = '#1f2937', '#6b7280', '#ffffff'
RED, BLUE, GREEN, AMBER, PURPLE, GRAY = '#e11d48', '#2563eb', '#16a34a', '#d97706', '#7c3aed', '#e5e7eb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.edgecolor': MUTED})


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis('off')
    return fig, ax


def fit_text(fig, ax, t, w, h, pad=0.14):
    """Shrink font size until the text bbox fits inside w x h (data units) with padding."""
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for _ in range(60):
        bb = t.get_window_extent(r)
        (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        if (x1 - x0) <= w - 2 * pad and (y1 - y0) <= h - 2 * pad * 0.8:
            return
        t.set_fontsize(t.get_fontsize() - 0.25)


def box(fig, ax, x, y, w, h, text, fc, ec=None, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.1', fc=fc, ec=ec or fc, lw=1.4))
    t = ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
                fontweight='bold' if bold else 'normal', linespacing=1.4)
    fit_text(fig, ax, t, w, h)


def arrow(ax, p, q, color=MUTED, gap=5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=13, lw=1.4, color=color, shrinkA=gap, shrinkB=gap))


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches='tight', pad_inches=0.25, facecolor=BG)
    plt.close(fig)


def architecture():
    fig, ax = canvas(15, 6.2, 15, 6.2)
    ax.text(7.5, 5.95, 'Detector architecture', ha='center', fontsize=14, fontweight='bold')
    box(fig, ax, 0.2, 2.45, 2.2, 1.3, 'Input image\n1024 x 768\n(never resized)', '#fef3c7', AMBER)
    box(fig, ax, 3.1, 3.7, 2.0, 1.1, '12 tiles\n256 x 256', '#fde68a', AMBER)
    box(fig, ax, 2.9, 0.75, 2.4, 1.8, 'Common CNN\nEfficientNet-B0\nfull image, fp32\nRGB + high-pass\nresidual (6 ch)', '#dcfce7', GREEN)
    box(fig, ax, 5.8, 3.5, 2.5, 1.5, 'DINOv3 ViT-L/16\nfrozen, 303M params\n3072 patch tokens', '#dbeafe', BLUE, bold=True)
    box(fig, ax, 9.0, 4.5, 2.3, 1.1, 'Gate\nattention pool + softmax', '#ede9fe', PURPLE)
    box(fig, ax, 9.0, 2.45, 2.3, 1.5, '24 Experts\ngate-weighted sum\n-> 64-d + score', '#ede9fe', PURPLE, bold=True)
    box(fig, ax, 11.9, 1.5, 1.5, 2.8, 'concat\n64 + 1 + 1280\n= 1345', GRAY, MUTED, bold=True)
    box(fig, ax, 13.8, 2.3, 1.1, 1.2, 'MLP\n256 -> 1\nsigmoid', '#fee2e2', RED, bold=True)
    for p_, q_ in [((2.4, 3.3), (3.1, 4.25)), ((2.4, 2.9), (2.9, 1.9)), ((5.1, 4.25), (5.8, 4.25)),
                   ((8.3, 4.7), (9.0, 5.0)), ((8.3, 3.9), (9.0, 3.2)), ((10.15, 4.5), (10.15, 3.95)),
                   ((11.3, 3.2), (11.9, 3.2)), ((5.3, 1.7), (11.9, 1.9)), ((13.4, 2.9), (13.8, 2.9))]:
        arrow(ax, p_, q_)
    ax.text(10.45, 4.2, 'mixing weights', fontsize=9, color=MUTED, ha='left', va='center')
    ax.text(14.35, 1.85, 'P(fake)', fontsize=12, ha='center', color=RED, fontweight='bold')
    ax.text(7.5, 0.1, '311M parameters   |   0.64 GB   |   152 s for 524 images on a Colab T4',
            ha='center', fontsize=10.5, color=MUTED)
    save(fig, 'architecture.png')


def pipeline():
    fig, ax = canvas(14.2, 7.2, 14.2, 7.2)
    ax.text(7, 6.95, 'Data pipeline: one real photo, many kinds of fakes (about 17k training images)',
            ha='center', fontsize=14, fontweight='bold')
    box(fig, ax, 0.2, 2.95, 2.2, 1.4, 'Real photos\n2,500 (AI Hub)\nbyte-exact', '#dcfce7', GREEN, bold=True)
    gens = [
        ('Full generation', 'FLUX.1-schnell, Z-Image, SDXL, RealVisXL,\nKandinsky 2.2, Infinity-2B, FLUX.2 klein'),
        ('Full style transfer', 'SDXL img2img, classic OpenCV styles'),
        ('Partial generation', 'LaMa, SDXL / SD1.5 / Kandinsky inpaint,\nFLUX.2 klein edit'),
        ('Partial style transfer', 'styled region pasted back onto the original'),
        ('Specified filters', 'blur, Gaussian noise, salt-pepper,\nJPEG recompression'),
        ('Composite', 'background swap, object paste, clone'),
        ('VAE reconstruction', 'FLUX VAE encode -> decode'),
    ]
    top, step, bh = 5.9, 0.82, 0.68
    for i, (name, how) in enumerate(gens):
        y = top - i * step
        box(fig, ax, 3.3, y, 2.5, bh, name, '#fee2e2', RED, bold=True)
        box(fig, ax, 6.1, y, 4.3, bh, how, '#f9fafb', MUTED, 9)
        arrow(ax, (2.4, 3.65), (3.3, y + bh / 2), '#cbd5e1', gap=3)
        arrow(ax, (10.4, y + bh / 2), (11.2, 3.65 + (3 - i) * 0.12), '#cbd5e1', gap=3)
    box(fig, ax, 11.2, 2.65, 2.7, 2.0, 'Normalize\ncenter crop 4:3 -> 1024x768\nJPEG q92, RGB, no EXIF\nsame for real and fake', '#fef3c7', AMBER, 9.5)
    ax.text(7, 0.2, 'Split by plant ID to avoid leakage.  Manifest, seed and prompt version recorded for every image.',
            ha='center', fontsize=10.5, color=MUTED)
    save(fig, 'pipeline.png')


def results():
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    names = ['exp02\nfirst full run', 'exp06\nlabel re-assign', 'exp11b (v3)\n24 Experts', 'v5\nthreshold 0.05', 'v7 (final)\n+ high-pass input']
    val = [0.874, 0.875, 0.885, 0.901, 0.902]
    lb = [0.77, 0.75, 0.83, 0.85, 0.87]
    x = list(range(5)); w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], val, w, color='#93c5fd', label='own validation')
    b2 = ax.bar([i + w / 2 for i in x], lb, w, color=BLUE, label='leaderboard (hidden 524 images)')
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f'{b.get_height():.3f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0.6, 0.96)
    ax.set_ylabel('accuracy'); ax.legend(loc='upper left', frameon=False, fontsize=9.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Own validation overestimates the hidden test', fontsize=12, fontweight='bold', pad=12)
    fig.text(0.5, 0.0, 'exp06 leaderboard value is an estimate', fontsize=8.5, color=MUTED, ha='center')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, 'results.png')


def experiments():
    items = [('exp08  fusion LayerNorm', 0.001), ('exp09  8 type-level Experts', -0.002),
             ('exp13  Common/Expert weighting', 0.004), ('exp15  top-k pooling', -0.010),
             ('exp16  high-pass residual input (in v7)', 0.003), ('exp17  feature dropout', -0.002),
             ('exp18  FFT spectrum features', 0.006), ('v5  decision threshold p>=0.05', 0.016)]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ys = list(range(len(items)))[::-1]
    ax.axvspan(-0.005, 0.005, color=GRAY, alpha=0.8, label='rough noise level (about 0.005, not a confidence interval)')
    for y, (n, d) in zip(ys, items):
        ax.barh(y, d, color=GREEN if n.startswith('v5') else (AMBER if n.startswith('exp16') else '#9ca3af'), height=0.6)
        ax.text(0.0215, y, f'{d:+.3f}', va='center', ha='right', fontsize=9.5,
                fontweight='bold' if n.startswith('v5') else 'normal', color=GREEN if n.startswith('v5') else INK)
    ax.set_yticks(ys); ax.set_yticklabels([n for n, _ in items], fontsize=9.5)
    ax.axvline(0, color=INK, lw=1); ax.set_xlim(-0.02, 0.0225); ax.set_ylim(-0.7, len(items) + 0.2)
    ax.set_xlabel('change in own validation metric vs. its baseline (accuracy)', fontsize=9.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GRAY, label='rough noise level (about 0.005, not a confidence interval)'), Patch(color=AMBER, label='idea used in the final model (v7)'), Patch(color=GREEN, label='decision-threshold shift')], loc='upper center', bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=9, ncol=1); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('On own validation, architecture ideas stayed within noise; the threshold shift gave the gain',
                 fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    save(fig, 'experiments.png')


def social_preview():
    """1280x640 card for GitHub Settings > Social preview (upload by hand; no API)."""
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12.8); ax.set_ylim(0, 6.4); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0, 0), 12.8, 6.4, boxstyle='square,pad=0', fc='#f8fafc', ec='none'))
    ax.add_patch(FancyBboxPatch((0, 0), 0.28, 6.4, boxstyle='square,pad=0', fc=RED, ec='none'))
    ax.text(0.8, 5.25, 'Strawberry Fake Detector', fontsize=40, fontweight='bold', color=INK, va='center')
    ax.text(0.8, 4.5, 'Real or AI-generated? DINOv3 + Mixture-of-Experts', fontsize=19, color=MUTED, va='center')
    stats = [('0.87', 'accuracy on the\nhidden 524-image test', GREEN),
             ('24', 'generator-specific\nExperts', PURPLE),
             ('152 s', '524 images on a\nColab T4 (limit 5 min)', BLUE)]
    for i, (big, small, c) in enumerate(stats):
        x = 0.8 + i * 3.85
        ax.add_patch(FancyBboxPatch((x, 2.0), 3.45, 1.95, boxstyle='round,pad=0,rounding_size=0.12', fc='white', ec=c, lw=2))
        ax.text(x + 0.3, 3.35, big, fontsize=34, fontweight='bold', color=c, va='center')
        ax.text(x + 0.3, 2.55, small, fontsize=13, color=INK, va='center', linespacing=1.35)
    chips = ['12 tiles', 'frozen DINOv3', 'Experts + Gate', 'P(fake)']
    x = 0.8
    for i, label in enumerate(chips):
        w = 0.15 * len(label) + 0.6
        ax.add_patch(FancyBboxPatch((x, 0.75), w, 0.6, boxstyle='round,pad=0,rounding_size=0.1', fc='#e0e7ff' if i < 3 else '#fee2e2', ec='none'))
        ax.text(x + w / 2, 1.05, label, fontsize=13, ha='center', va='center', color=INK, fontweight='bold')
        x += w
        if i < len(chips) - 1:
            ax.text(x + 0.2, 1.05, '>', fontsize=16, ha='center', va='center', color=MUTED)
            x += 0.4
    ax.text(12.4, 0.35, 'github.com/baejh3333-del/strawberry-fake-detector', fontsize=11, color=MUTED, ha='right', va='center')
    fig.savefig(OUT / 'social-preview.png', dpi=100, facecolor='#f8fafc')
    plt.close(fig)


if __name__ == '__main__':
    architecture(); pipeline(); results(); experiments(); social_preview(); print('saved')
