import math
import numpy as np

def composite_strain_at_fibre(q, R, layer_props, z_fibre_from_ref):
    """
    q: uniform pressure (Pa) = rho*g*h
    R: plate radius (m)
    layer_props: list of dicts, each with keys:
        { "t": thickness (m), "E": Young's modulus (Pa), "nu": Poisson's ratio }
      Order/layer stacking is arbitrary as long as z positions are computed from same ref.
    z_fibre_from_ref: z coordinate (m) of fibre location measured from same reference (e.g. bottom)
    
    Returns: (w0, eps_fibre) where
      w0 = center deflection (m),
      eps_fibre = radial strain at fibre (dimensionless, not microstrain).
    """
    # Per-unit-width areas and centroids
    A = []
    z_centroids = []
    Estar = []  # E/(1-nu^2)
    for i,layer in enumerate(layer_props):
        t = layer["t"]
        E = layer["E"]
        nu = layer.get("nu", 0.3)
        # centroid of layer (assume stack built bottom->top in layer_props)
        if i == 0:
            z_bottom = 0.0
        else:
            z_bottom = sum(lp["t"] for lp in layer_props[:i])
        zc = z_bottom + t/2.0
        A.append(t)                 # per-unit-width area
        z_centroids.append(zc)
        Estar.append(E / (1.0 - nu**2))

    A = np.array(A)
    zc = np.array(z_centroids)
    Estar = np.array(Estar)

    # Neutral axis:
    z_na = np.sum(Estar * A * zc) / np.sum(Estar * A)

    # Composite bending stiffness D (Pa * m^3)
    Ic = (A * ( (A)**2 ) )*0  # placeholder, we compute using t^3/12 below
    # compute I_c,i = t_i^3/12 per unit width
    Ics = np.array([ (layer_props[i]["t"]**3)/12.0 for i in range(len(layer_props)) ])
    D = np.sum(Estar * (Ics + A * (zc - z_na)**2))

    # centre deflection
    w0 = q * R**4 / (64.0 * D)

    # fibre z relative to NA
    z_f_rel = z_fibre_from_ref - z_na

    # radial strain at fibre (dimensionless)
    eps_fibre = (z_f_rel * q * R**2) / (16.0 * D)

    return w0, eps_fibre

# Example usage: diaphragm layer (steel-like) + epoxy layer (bottom)
rho = 1000.0
g = 9.81
h = 0.1          # fluid height (m)
q = rho * g * h   # Pa

R = 0.104          # plate radius (m)

layers = [
    {"t": 0.00012, "E": 200e9, "nu": 0.285},   # main diaphragm (top)
    {"t": 0.00082, "E": 3.4e9, "nu": 0.35}   # epoxy (bottom)
]
# fibre sits inside epoxy near bottom surface; if reference is bottom of stack:
z_fibre = 0.00015  # e.g. 0.25 mm above bottom reference

w0, eps = composite_strain_at_fibre(q, R, layers, z_fibre)
print("w0 (m):", w0)
print("eps (microstrain):", eps * 1e6)
