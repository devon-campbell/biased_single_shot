#!/usr/bin/env python3

import yaml
import numpy as np 
import galois 
import datetime
import time 
import graphs 
import peg_utils
import os
from tqdm import tqdm
import math

def two_point_lam(n, m, rho_node):
    deg_CN = sum((i+1)*w for i, w in enumerate(rho_node))
    d_avg  = (m * deg_CN) / n
    d_hi   = math.ceil(d_avg)
    d_lo   = d_hi - 1
    w_hi   = d_avg - d_lo
    w_lo   = 1 - w_hi
    lam    = [0.0] * d_hi
    lam[d_lo - 1] = w_lo
    lam[d_hi - 1] = w_hi
    return np.array(lam)

def two_point_rho(n, m, lam_node, epsilon=1e-4):
    d_avg_vn = sum((i + 1) * w for i, w in enumerate(lam_node))
    d_avg_cn = (n * d_avg_vn) / m

    d_hi = math.ceil(d_avg_cn)
    d_lo = d_hi - 1
    w_hi = d_avg_cn - d_lo
    w_lo = 1 - w_hi

    rho = [0.0] * d_hi
    rho[d_lo - 1] = w_lo
    rho[d_hi - 1] = w_hi

    # Clean up tiny entries
    rho = np.array(rho)
    rho[rho < epsilon] = 0.0

    # Renormalize to ensure sum = 1
    total = rho.sum()
    if not np.isclose(total, 1.0):
        rho = rho / total

    return np.trim_zeros(rho, 'b')

#Load settings
with open("config.yml", "r") as ymlfile:
    settings = yaml.safe_load(ymlfile)
    lam_node_cfg = settings["lam_node"]
    rho_node_cfg = settings["rho_node"]

m, n, scaling_factor = settings["m"], settings["n"], settings["scaling_factor"]
print(lam_node_cfg, type(lam_node_cfg))
print(rho_node_cfg, type(rho_node_cfg))

if lam_node_cfg is not None:
    print('Building rho_node')
    lam_node =  np.array(lam_node_cfg)
    rho_node = two_point_rho(n, m, lam_node_cfg, epsilon=1e-4)
elif rho_node_cfg is not None:
    print('Building lam_node')
    rho_node =  np.array(rho_node_cfg)
    lam_node = two_point_lam(n, m, rho_node_cfg)

header = f"""
===================================================================
Creating ({m},{n},{scaling_factor}) code with the {settings["girth_search_depth"]}-edge QC-PEGA algorithm for 
variable node edge distribution.
lam_node: {lam_node}
rho_node: {rho_node}
Output file: {settings["output_fname"]}
===================================================================
"""
print(header)
G = graphs.QC_tanner_graph(m, n, scaling_factor)

# Get the node perspective degree distributions from density evolution optimization
vn_degrees = np.flip(peg_utils.to_degree_distribution(lam_node,G.n_vn))
cn_degrees = peg_utils.to_degree_distribution(rho_node,G.n_cn)

if not settings["seed"] == 'None':
    np.random.seed(int(settings["seed"]))
t0 = time.time()

for current_vn_index in tqdm(range(0, G.n_vn, G.N), desc="Building Tanner graph"):
    d = vn_degrees[current_vn_index]
    for k in range(1, int(d+1)):
        rk = int(min(settings["girth_search_depth"], d - k +1))
        max_girth, cn_girths = peg_utils.rk_edge_local_girth(G, current_vn_index, rk, cn_degrees)
    
        ci = peg_utils.strategy1(max_girth, cn_girths, G, current_vn_index, cn_degrees)
        G.add_cyclical_edge_set(ci, current_vn_index)

    dt = time.time() - t0
    completed_progress = (current_vn_index+G.N)/(G.n_vn)
    time_left = dt/completed_progress - dt        
    status = f"{100*completed_progress:.2f}% completed, elapsed time {int(dt//60)}:{dt% 60:.2f}s. Approximate time left: {int(time_left//60)}:{time_left % 60:.2f}"
    print(status, end='\r', flush=True)
    

print("")
print(f"Edge growth finsihed. Total elapsed time: {int(dt//60)} minutes, {dt % 60:.2f} seconds.")

print(np.bincount(G.get_check_degrees()))
print(np.bincount(cn_degrees))
print(np.bincount(G.get_var_degrees()))
print(np.bincount(vn_degrees))

G_reordered = peg_utils.make_invertable(G)
print("Matrix invertion sucessfull, saving file...")

metadata = f"""
# QC-code generated with the r-edge metric constrained QC-PEG algorithm.
# 
# Name   : {settings["output_fname"]}
# R      : {settings["girth_search_depth"]}
# Metric : Minimum distance
# Date   : {datetime.datetime.now()}
"""
os.makedirs('data/', exist_ok=True)
G_reordered.save(settings["output_fname"])
f = open(settings["output_fname"], 'a')
f.write(metadata)
f.close()

print("MM-QC-PEGA completed.")

peg_utils.graph_stats(G,settings["output_fname"])