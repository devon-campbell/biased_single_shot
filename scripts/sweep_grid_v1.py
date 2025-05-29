from itertools import product

error_rate      = [0.0, .01, .02, .04, .06, .08]
p_meas_err      = [0.0, .01, .02, .04]
bias_vals       = [5, 10, 50, 100, 1000]
hadamard_rotate = [True, False]
run_ss          = [True, False]
apply_ss        = [True, False]
channel_update  = [None]
xyz_bias        = [[1, 1, 1]] + [[1, 1, b] for b in bias_vals] + [[b, 1, 1] for b in bias_vals]


# Build full grid with all combinations
full_grid = list(product(error_rate, p_meas_err, xyz_bias, hadamard_rotate, run_ss, apply_ss, channel_update))

# Filter out cases where run_ss == False and apply_ss == True
grid = [
    (err, meas, bias, hadamard, ss, apply, chan)
    for (err, meas, bias, hadamard, ss, apply, chan) in full_grid
    if not (ss is False and apply is True)
]

print(f"{len(grid)=:,}")
