import json
import math
import os
import matplotlib.pyplot as plt
 
# ── Load data ──
base      = os.path.dirname(__file__)
data_path = os.path.join(base, 'run_data.json')
 
with open(data_path, 'r') as f:
    state_log = json.load(f)
 
print(f"Loaded {len(state_log)} entries from run_data.json")
 
TARGET_DIST = 0.5
 
# ── Extract data ──
times  = [s['t']             for s in state_log]
costs  = [s['cumul_cost']    for s in state_log]
dists  = [s['dist']          for s in state_log]
errors = [math.degrees(s['heading_error']) for s in state_log]
rob_x  = [s['pos'][0]        for s in state_log]
rob_y  = [s['pos'][1]        for s in state_log]
tgt_x  = [s['target_pos'][0] for s in state_log]
tgt_y  = [s['target_pos'][1] for s in state_log]
 
# ── Plot 1: time series ──
fig1, axes = plt.subplots(3, 1, figsize=(10, 8))
fig1.suptitle('Run Results', fontsize=14)
 
axes[0].plot(times, dists, 'b')
axes[0].axhline(TARGET_DIST, color='r', linestyle='--', label='target')
axes[0].set_ylabel('Distance'); axes[0].legend(); axes[0].grid(True)
 
axes[1].plot(times, errors, 'm')
axes[1].axhline(0, color='r', linestyle='--')
axes[1].set_ylabel('Heading Error (deg)'); axes[1].grid(True)
 
axes[2].plot(times, costs, 'g')
axes[2].set_ylabel('Cumulative Cost'); axes[2].grid(True)
axes[2].set_xlabel('Time (s)')
 
plt.tight_layout()
fig1.savefig(os.path.join(base, 'run_timeseries.png'), dpi=150, bbox_inches='tight')
 
# ── Plot 2: top-down path ──
fig2, ax = plt.subplots(1, 1, figsize=(8, 8))
fig2.suptitle('Top-down path view', fontsize=14)
 
ax.plot(rob_x, rob_y, 'b', label='robot',  linewidth=1.5)
ax.plot(tgt_x, tgt_y, 'r', label='target', linewidth=1.5)
ax.plot(rob_x[0], rob_y[0], 'bo', markersize=8, label='robot start')
ax.plot(tgt_x[0], tgt_y[0], 'ro', markersize=8, label='target start')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.legend(); ax.grid(True); ax.set_aspect('equal')
 
plt.tight_layout()
fig2.savefig(os.path.join(base, 'run_path.png'), dpi=150, bbox_inches='tight')

# ── Plot 3: X and Y positions over time ──
fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6))
fig3.suptitle('Robot vs Target Position Over Time', fontsize=14)

axes3[0].plot(times, rob_x, 'b', label='robot x', linewidth=1.5)
axes3[0].plot(times, tgt_x, 'r', label='target x', linewidth=1.5)
axes3[0].set_ylabel('X position'); axes3[0].legend(); axes3[0].grid(True)

axes3[1].plot(times, rob_y, 'b', label='robot y', linewidth=1.5)
axes3[1].plot(times, tgt_y, 'r', label='target y', linewidth=1.5)
axes3[1].set_ylabel('Y position'); axes3[1].legend(); axes3[1].grid(True)
axes3[1].set_xlabel('Time (s)')

plt.tight_layout()
fig3.savefig(os.path.join(base, 'run_xy.png'), dpi=150, bbox_inches='tight')

plt.show()