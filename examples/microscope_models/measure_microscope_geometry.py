# Auto-generated from: examples/microscope_models/measure_microscope_geometry.ipynb
# Conversion keeps code cells and comments out notebook magics/shell lines.

# %% [cell 1 - code]
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib widget

# %% [cell 2 - code]
# Manually fill y-pixel positions for each lens label.
# Leave as None until you set a value.
label_y = {
    "Source": 250,
    "CL1": 340,
    "CL2": 360,
    "CLMini": 529,
    "Objective lens Pre": 570,
    "Sample": 575,
    "Objective lens Post": 576,
    'SAAperture': 666,
    "IL1": 672,
    "IL2": 709,
    "IL3": 739,
    "PL1": 764,
    "Screen": 920,
}

length_scale_px = 624 - 528
length_scale_m = 0.2  # 20 cm - length of elliptical box on left hand side of image.
px_to_m = length_scale_m / length_scale_px

label_y_m = {k: (v - 575) * px_to_m if v is not None else None for k, v in label_y.items()}
print(label_y_m)

img = plt.imread('microscope.png')
# Plot image with horizontal label lines at each filled y value
fig, ax = plt.subplots(figsize=(7, 11))
ax.imshow(img, origin="upper")

for name, y in label_y.items():
    if y is None:
        continue
    y = int(y)
    ax.axhline(y, color="lime", lw=1.5, alpha=0.9)
    ax.text(
        8, y - 4, f"{name} (y={y})",
        color="lime", fontsize=10, fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2),
    )

ax.set_title("Manual lens y-labels")
ax.axis("off")
plt.show()

# Optional cleaned dict (only filled labels)
lens_z = {k: int(v) for k, v in label_y.items() if v is not None}
lens_z

y_vals_m = np.array([v for v in label_y_m.values() if v is not None], dtype=float)
distances = np.diff(y_vals_m - np.min(y_vals_m))
print("Distances between lenses (in meters):", distances)


def _abs_delta_m(a: str, b: str) -> float:
    va = label_y_m.get(a, None)
    vb = label_y_m.get(b, None)
    if va is None or vb is None:
        raise ValueError(f"Missing y-label(s) for '{a}' or '{b}'.")
    return float(abs(vb - va))

CAD_PRIOR_POS_M = {
    'Source': label_y_m['Source'],
    'CL1': label_y_m['CL1'],
    'CL2': label_y_m['CL2'],
    'CLMini': label_y_m['CLMini'],
    'Objective lens Pre': label_y_m['Objective lens Pre'],
    'Sample': label_y_m['Sample'],
    'Objective lens Post': label_y_m['Objective lens Post'],
    'SAAperture': label_y_m['SAAperture'],
    'IL1': label_y_m['IL1'],
    'IL2': label_y_m['IL2'],
    'IL3': label_y_m['IL3'],
    'PL1': label_y_m['PL1'],
    'Screen': label_y_m['Screen'],
}

# CAD-derived geometric priors used by projector fit (source/objective/post-obj onwards).
CAD_PRIOR_DISTS_M = {
    'd_object_to_opl': _abs_delta_m('Sample', 'Objective lens Post'),
    'd_opl_to_saa': _abs_delta_m('Objective lens Post', 'SAAperture'),
    'd_opl_to_il1': _abs_delta_m('Objective lens Post', 'IL1'),
    'd_il1_to_il2': _abs_delta_m('IL1', 'IL2'),
    'd_il2_to_il3': _abs_delta_m('IL2', 'IL3'),
    'd_il3_to_pl1': _abs_delta_m('IL3', 'PL1'),
    'd_pl1_to_det': _abs_delta_m('PL1', 'Screen'),
}

print('CAD priors [mm]:')
for k, v in CAD_PRIOR_DISTS_M.items():
    print(f"  {k}: {1e3 * v:.3f}")
