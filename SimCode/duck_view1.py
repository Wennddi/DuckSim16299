
import pybullet as p
import pybullet_data
import time
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
#  CONNECT
# ─────────────────────────────────────────────────────────────────────────────
try:
    p.disconnect()
except:
    pass

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

base      = os.path.dirname(__file__)
duck_path = os.path.join(base, "duck.urdf")
robot     = p.loadURDF(duck_path, [0, 0, 0.27])

spawn_angle  = random.uniform(0, 2 * math.pi)
spawn_dist   = 5.0
target_duck  = p.loadURDF("duck_vhacd.urdf", [
    math.cos(spawn_angle) * spawn_dist,
    math.sin(spawn_angle) * spawn_dist,
    0.15
], globalScaling=10.0)

for i in range(-1, p.getNumJoints(robot)):
    p.changeDynamics(robot, i, lateralFriction=20.0, linearDamping=0.9, angularDamping=0.9)

# p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

p.resetDebugVisualizerCamera(
    cameraDistance=5, cameraYaw=45, cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

# ─────────────────────────────────────────────────────────────────────────────
#  GAIT PARAMS
# ─────────────────────────────────────────────────────────────────────────────
SWING       = 0.2
SUPPORT     = 0.05
PERIOD      = 0.6
TARGET_DIST = 0.5
WADDLE_AMOUNT = 0.08

# ─────────────────────────────────────────────────────────────────────────────
#  PID CLASS  (your original, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit   = limit
        self.prev_error = 0
        self.integral   = 0

    def update(self, error, dt):
        self.integral  += error * dt
        derivative      = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.limit:
            output = max(-self.limit, min(self.limit, output))
        return output

    def reset(self):
        self.integral   = 0
        self.prev_error = 0

# ─────────────────────────────────────────────────────────────────────────────
#  COST FUNCTION  (from Value Functions lecture)
#  L(x, u) = w1*heading_error^2 + w2*dist_error^2 + w3*(turn^2 + speed^2)
#  Total cost C(π, x0) = Σ L(xi, π(xi))  over the run
# ─────────────────────────────────────────────────────────────────────────────
W_HEADING = 1.0    # penalise not facing the target
W_DIST    = 2.0    # penalise being far from target
W_EFFORT  = 0.05   # penalise large control actions (energy cost)

cumulative_cost = 0.0
step_count      = 0

def compute_cost(heading_error, dist_error, turn, speed):
    return (W_HEADING * heading_error**2 +
            W_DIST    * dist_error**2    +
            W_EFFORT  * (turn**2 + speed**2))

# ─────────────────────────────────────────────────────────────────────────────
#  STATE LOGGER  — records full state vector each step
#  State: [t, pos_x, pos_y, yaw, dist, heading_error, turn, speed, step_cost]
# ─────────────────────────────────────────────────────────────────────────────
state_log = []
LOG_EVERY = 240   # log once per simulated second

# ─────────────────────────────────────────────────────────────────────────────
#  GUI SLIDERS  — live gain tuning (connects to spring-damper intuition)
#  Kp acts like spring stiffness k: higher = faster response, more oscillation
#  Kd acts like damper b:           higher = smoother, slower
# ─────────────────────────────────────────────────────────────────────────────
kp_h_slider = p.addUserDebugParameter("Heading Kp  (spring)", 0.0, 5.0, 1.0)
kd_h_slider = p.addUserDebugParameter("Heading Kd  (damper)", 0.0, 2.0, 0.3)
kp_d_slider = p.addUserDebugParameter("Distance Kp (spring)", 0.0, 2.0, 0.5)
kd_d_slider = p.addUserDebugParameter("Distance Kd (damper)", 0.0, 1.0, 0.1)

heading_pid  = PID(kp=1.0, ki=0.05, kd=0.3, limit=0.6)
distance_pid = PID(kp=0.5, ki=0.05, kd=0.1, limit=0.4)

# ─────────────────────────────────────────────────────────────────────────────
#  WANDERING TARGET  (your original logic, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
wander_speed    = 0.15
wander_dir      = [1.0, 0.0]
wander_timer    = 0.0
wander_interval = 80.0

MIN_SPEED = 0.1
MAX_SPEED = 0.30

def new_wander_dir():
    angle = np.random.uniform(0, 2 * math.pi)
    return [math.cos(angle), math.sin(angle)]

def new_wander_speed():
    return np.random.uniform(MIN_SPEED, MAX_SPEED)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
t    = 0.0
dt   = 1 / 240
yaw  = 0.0

try: 
    while True:
        t            += dt
        wander_timer += dt
        step_count   += 1

        # ── Read sliders and update PID gains live ──
        heading_pid.kp  = p.readUserDebugParameter(kp_h_slider)
        heading_pid.kd  = p.readUserDebugParameter(kd_h_slider)
        distance_pid.kp = p.readUserDebugParameter(kp_d_slider)
        distance_pid.kd = p.readUserDebugParameter(kd_d_slider)

        # ── Wandering target movement (your original logic) ──
        if wander_timer >= wander_interval:
            wander_dir   = new_wander_dir()
            wander_speed = new_wander_speed()
            wander_timer = 0.0

        target_pos, _ = p.getBasePositionAndOrientation(target_duck)
        new_target_pos = [
            target_pos[0] + wander_dir[0] * wander_speed * dt,
            target_pos[1] + wander_dir[1] * wander_speed * dt,
            0.15
        ]

        target_yaw_desired = math.atan2(wander_dir[1], wander_dir[0]) + math.pi
        _, current_target_orn = p.getBasePositionAndOrientation(target_duck)
        current_target_euler  = p.getEulerFromQuaternion(current_target_orn)
        current_target_yaw    = current_target_euler[2]
        yaw_diff     = (target_yaw_desired - current_target_yaw + math.pi) % (2 * math.pi) - math.pi
        new_target_yaw = current_target_yaw + yaw_diff * min(1.0, 2.0 * dt)
        target_orn   = p.getQuaternionFromEuler([math.pi / 2, 0, new_target_yaw])
        p.resetBasePositionAndOrientation(target_duck, new_target_pos, target_orn)

        # ── GET STATE  x = [pos, yaw, velocity] ──
        robot_pos, robot_orn     = p.getBasePositionAndOrientation(robot)
        robot_vel, robot_ang_vel = p.getBaseVelocity(robot)
        euler = p.getEulerFromQuaternion(robot_orn)
        yaw   = euler[2]

        # Velocity in world frame (for cost and state logging)
        vel_x, vel_y = robot_vel[0], robot_vel[1]
        speed_actual = math.sqrt(vel_x**2 + vel_y**2)

        # ── COMPUTE ERRORS ──
        dx       = target_pos[0] - robot_pos[0]
        dy       = target_pos[1] - robot_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)

        heading_error = (angle_to_target - yaw + math.pi) % (2 * math.pi) - math.pi
        dist_error    = distance - TARGET_DIST

        # ── PID CONTROL  u = [turn, speed] ──
        turn  = heading_pid.update(heading_error, dt)
        speed = distance_pid.update(dist_error, dt) if abs(dist_error) >= 0.05 else 0.0

        # ── COST  L(x, u) at this timestep ──
        step_cost        = compute_cost(heading_error, dist_error, turn, speed)
        cumulative_cost += step_cost * dt

        # ── STATE LOG  (one entry per second) ──
        if step_count % LOG_EVERY == 0:
            state_log.append({
                't':             t,
                'pos':           (robot_pos[0], robot_pos[1]),
                'target_pos':    (target_pos[0], target_pos[1]),
                'yaw':           yaw,
                'dist':          distance,
                'heading_error': heading_error,
                'turn':          turn,
                'speed':         speed,
                'step_cost':     step_cost,
                'cumul_cost':    cumulative_cost,
            })
            print(f"t={t:6.1f}s | dist={distance:.2f} | "
                f"hdg_err={math.degrees(heading_error):+.1f}° | "
                f"turn={turn:+.3f} spd={speed:.3f} | "
                f"cost={step_cost:.4f} cumul={cumulative_cost:.2f}")

        # ── Apply motion ──
        yaw += turn * dt
 
        # ── GAIT + WADDLE ──
        phase        = int((t % PERIOD) < (PERIOD / 2))
        forward_bias = max(0.0, min(0.3, speed))
 
        if speed > 0.01:
            if phase == 0:
                left_angle   =  SUPPORT + forward_bias + turn * 0.3
                right_angle  = -SWING   + forward_bias - turn * 0.3
                waddle_roll  =  WADDLE_AMOUNT   # lean right on right stance
            else:
                left_angle   =  SWING   + forward_bias + turn * 0.3
                right_angle  = -SUPPORT + forward_bias - turn * 0.3
                waddle_roll  = -WADDLE_AMOUNT   # lean left on left stance
        else:
            left_angle  =  0.05
            right_angle = -0.05
            waddle_roll =  0.0   # stand straight when idle
 
        # apply yaw + waddle roll together — replaces the old [0, 0, yaw]
        locked_orn = p.getQuaternionFromEuler([waddle_roll, 0, yaw])
 
        p.resetBasePositionAndOrientation(robot, robot_pos, locked_orn)
 
        forward_x = math.cos(yaw) * speed * dt
        forward_y = math.sin(yaw) * speed * dt
        new_robot_pos = [robot_pos[0] + forward_x,
                         robot_pos[1] + forward_y,
                         robot_pos[2]]
        p.resetBasePositionAndOrientation(robot, new_robot_pos, locked_orn)
 
        p.setJointMotorControl2(robot, 0, p.POSITION_CONTROL, left_angle,  force=50)
        p.setJointMotorControl2(robot, 2, p.POSITION_CONTROL, right_angle, force=50)
 
        p.stepSimulation()
        time.sleep(dt)
    
finally: 
    import json
    import subprocess
    with open(os.path.join(base, 'run_data.json'), 'w') as f:
        json.dump(state_log, f)
    print(f"Saved {len(state_log)} entries to run_data.json")
    subprocess.Popen(['python', os.path.join(base, 'plot_results.py')])


    # runs when you close the window or hit Ctrl+C
    # times  = [s['t']           for s in state_log]
    # costs  = [s['cumul_cost']  for s in state_log]
    # dists  = [s['dist']        for s in state_log]
    # errors = [math.degrees(s['heading_error']) for s in state_log]

    # rob_x  = [s['pos'][0]        for s in state_log]
    # rob_y  = [s['pos'][1]        for s in state_log]
    # tgt_x  = [s['target_pos'][0] for s in state_log]
    # tgt_y  = [s['target_pos'][1] for s in state_log]

    # fig, axes = plt.subplots(4, 1, figsize=(10, 13))

    # axes[0].plot(times, dists,  'b')
    # axes[0].axhline(TARGET_DIST, color='r', linestyle='--', label='target')
    # axes[0].set_ylabel('Distance'); axes[0].legend(); axes[0].grid(True)

    # axes[1].plot(times, errors, 'm')
    # axes[1].axhline(0, color='r', linestyle='--')
    # axes[1].set_ylabel('Heading Error (deg)'); axes[1].grid(True)

    # axes[2].plot(times, costs,  'g')
    # axes[2].set_ylabel('Cumulative Cost'); axes[2].grid(True)
    # axes[2].set_xlabel('Time (s)')

    # axes[3].plot(rob_x, rob_y, 'b', label='robot',  linewidth=1.5)
    # axes[3].plot(tgt_x, tgt_y, 'r', label='target', linewidth=1.5)
    # axes[3].plot(rob_x[0], rob_y[0], 'bo', markersize=8)
    # axes[3].plot(tgt_x[0], tgt_y[0], 'ro', markersize=8)
    # axes[3].set_xlabel('X position')
    # axes[3].set_ylabel('Y position')
    # axes[3].set_title('Top-down path view')
    # axes[3].legend(); axes[3].grid(True); axes[3].set_aspect('equal')

    # plt.tight_layout()
    # plt.show()


