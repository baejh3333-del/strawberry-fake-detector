"""Regenerates the README figures (no dataset images are used): python assets/make_figures.py"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent
INK, MUTED, BG = '#1f2937', '#6b7280', '#ffffff'
RED, BLUE, GREEN, AMBER, PURPLE, GRAY = '#e11d48', '#2563eb', '#16a34a', '#d97706', '#7c3aed', '#e5e7eb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK, 'axes.edgecolor': MUTED})


def box(ax, x, y, w, h, text, fc, ec=None, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec or fc, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold' if bold else 'normal', linespacing=1.35)


def arrow(ax, p, q, color=MUTED):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14, lw=1.5, color=color, shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(13, 5.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis('off')
    box(ax, 0.2, 2.2, 1.9, 1.2, 'Input image\n1024 x 768\n(never resized)', '#fef3c7', AMBER)
    box(ax, 2.7, 3.4, 1.9, 1.0, '12 tiles\n256 x 256', '#fde68a', AMBER)
    box(ax, 2.7, 0.9, 1.9, 1.3, 'Common CNN\nEfficientNet-B0\n(full image, fp32)', '#dcfce7', GREEN)
    box(ax, 5.2, 3.2, 2.2, 1.4, 'DINOv3 ViT-L/16\nfrozen, 303M params\n3072 patch tokens', '#dbeafe', BLUE, bold=True)
    box(ax, 7.9, 4.25, 2.1, 0.9, 'Gate\nattention pool\n+ softmax', '#ede9fe', PURPLE, 9)
    box(ax, 8.0, 2.55, 1.9, 1.3, '24 Experts\nattention pool\n-> 64-d + score', '#ede9fe', PURPLE, bold=True)
    box(ax, 10.4, 2.2, 1.2, 1.5, 'concat\n64+1+1280\n= 1345', GRAY, MUTED, 9)
    box(ax, 11.9, 2.35, 1.0, 1.2, 'MLP\n256 -> 1\nsigmoid', '#fee2e2', RED, 9, bold=True)
    for p, q in [((2.1, 3.0), (2.7, 3.7)), ((2.1, 2.6), (2.7, 1.8)), ((4.6, 3.9), (5.2, 3.9)), ((7.4, 4.2), (8.0, 4.7)),
                 ((7.4, 3.6), (8.0, 3.2)), ((9.9, 4.7), (10.4, 3.5)), ((9.9, 3.2), (10.4, 3.0)), ((4.6, 1.55), (10.4, 2.5)),
                 ((11.6, 2.95), (11.9, 2.95))]:
        arrow(ax, p, q)
    ax.text(11.0, 4.45, 'mixing weights', fontsize=8, color=MUTED, ha='center')
    ax.text(12.4, 1.95, 'P(fake)', fontsize=11, ha='center', color=RED, fontweight='bold')
    ax.text(6.5, 5.4, 'Detector architecture: tile-level DINOv3 features + generator-specific Experts + full-image CNN',
            ha='center', fontsize=12, fontweight='bold')
    ax.text(6.5, 0.3, '311M parameters  |  0.64 GB  |  152 s for 524 images on a Colab T4', ha='center', fontsize=10, color=MUTED)
    save(fig, 'architecture.png')


def pipeline():
    fig, ax = plt.subplots(figsize=(13, 6.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.7); ax.axis('off')
    box(ax, 0.2, 2.5, 1.9, 1.2, 'Real photos\n2,500 (AI Hub)\nbyte-exact', '#dcfce7', GREEN, bold=True)
    gens = [
        ('Full generation', 'FLUX.1-schnell / Z-Image / SDXL / RealVisXL\nKandinsky 2.2 / Infinity-2B / FLUX.2 klein'),
        ('Full style transfer', 'SDXL img2img / classic OpenCV styles'),
        ('Partial generation', 'LaMa / SDXL, SD1.5, Kandinsky inpaint / klein edit'),
        ('Partial style transfer', 'styled region pasted back onto the original'),
        ('Specified filters', 'blur / Gaussian noise / salt-pepper / JPEG recompression'),
        ('Composite', 'background swap / object paste / clone'),
        ('VAE reconstruction', 'FLUX VAE encode -> decode'),
    ]
    for i, (name, how) in enumerate(gens):
        y = 5.35 - i * 0.78
        box(ax, 3.2, y, 2.3, 0.62, name, '#fee2e2', RED, 9, bold=True)
        box(ax, 5.7, y, 4.2, 0.62, how, '#f9fafb', MUTED, 8)
        arrow(ax, (2.1, 3.1), (3.2, y + 0.31), '#cbd5e1')
        arrow(ax, (9.9, y + 0.31), (10.4, 3.1), '#cbd5e1')
    box(ax, 10.4, 2.3, 2.4, 1.6, 'Normalize\ncenter crop 4:3 -> 1024x768\nJPEG q92, RGB, no EXIF\n(same for real & fake)', '#fef3c7', AMBER, 9)
    ax.text(6.5, 6.45, 'Data pipeline: one real photo -> many kinds of fakes (about 17k training images)',
            ha='center', fontsize=12, fontweight='bold')
    ax.text(6.5, 0.15, 'Split by plant ID to avoid leakage. Manifest + seed + prompt version recorded for every image.',
            ha='center', fontsize=9.5, color=MUTED)
    save(fig, 'pipeline.png')


def results():
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ['exp02\nfirst full run', 'exp06\nlabel re-assign', 'exp11b (v3)\n24 Experts', 'v5 (final)\nthreshold 0.05']
    val = [0.874, 0.875, 0.885, 0.901]
    lb = [0.77, 0.75, 0.83, 0.85]
    x = list(range(4)); w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], val, w, color='#93c5fd', label='own validation')
    b2 = ax.bar([i + w / 2 for i in x], lb, w, color=BLUE, label='leaderboard (hidden 524 images)')
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f'{b.get_height():.3f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0.6, 0.95)
    ax.set_ylabel('accuracy'); ax.legend(loc='upper left', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Own validation overestimates the hidden test', fontsize=11, fontweight='bold')
    fig.text(0.5, -0.02, 'exp06 leaderboard value is an estimate', fontsize=8, color=MUTED, ha='center')
    save(fig, 'results.png')


def experiments():
    items = [('exp08  fusion LayerNorm', 0.001), ('exp09  8 type-level Experts', -0.002),
             ('exp13  Common/Expert weighting', 0.004), ('exp15  top-k pooling', -0.010),
             ('exp16  high-pass residual input', 0.003), ('exp17  feature dropout', -0.002),
             ('exp18  FFT spectrum features', 0.006), ('v5  decision threshold p>=0.05', 0.016)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ys = list(range(len(items)))[::-1]
    ax.axvspan(-0.005, 0.005, color=GRAY, alpha=0.8, label='seed-to-seed noise (about 0.005)')
    for y, (n, d) in zip(ys, items):
        ax.barh(y, d, color=GREEN if n.startswith('v5') else '#9ca3af', height=0.6)
        ax.text(d + (0.0007 if d >= 0 else -0.0007), y, f'{d:+.3f}', va='center',
                ha='left' if d >= 0 else 'right', fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels([n for n, _ in items], fontsize=9)
    ax.axvline(0, color=INK, lw=1); ax.set_xlim(-0.02, 0.024)
    ax.set_xlabel('change in own validation metric vs. its baseline (accuracy)')
    ax.legend(loc='upper left', frameon=False, fontsize=9); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Seven architecture ideas stayed within noise; the decision rule gave the gain', fontsize=11, fontweight='bold')
    save(fig, 'experiments.png')


if __name__ == '__main__':
    architecture(); pipeline(); results(); experiments(); print('saved to', OUT)
