# Numerical safety constants used across TF and NumPy code paths.

SAT_MIN = 1e-9         # minimum saturation scale
HALF_LIFE_MIN = 1e-6   # minimum half-life before converting to delta
SIGMA_MIN = 1e-9       # minimum sigma guard when computing log-likelihoods
