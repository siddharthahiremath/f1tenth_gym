"""
Headless benchmark for the example planner scripts in this repo.

Runs each planner without rendering (so results aren't bottlenecked or
paced by the GUI) and reports the simulated lap time(s) achieved, plus how
long the benchmark itself took to compute (real elapsed time).

Usage:
    python3 benchmark_laptime.py
"""
import time
from argparse import Namespace

import gym
import numpy as np
import yaml
from f110_gym.envs.base_classes import Integrator

from mpc_planner import MPCPlanner
from waypoint_follow import PurePursuitPlanner

MAX_SIM_TIME = 120.0  # safety cutoff in case a planner never finishes a lap


def make_env(conf):
    return gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                     num_agents=1, timestep=0.01, integrator=Integrator.RK4)


def load_conf(config_path='config_example_map.yaml'):
    with open(config_path) as f:
        return Namespace(**yaml.load(f, Loader=yaml.FullLoader))


def benchmark_pure_pursuit(conf):
    work = {'tlad': 0.82461887897713965, 'vgain': 1.375}
    planner = PurePursuitPlanner(conf, 0.17145 + 0.15875)
    env = make_env(conf)

    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

    sim_time, steps = 0.0, 0
    start = time.time()
    while not done and sim_time < MAX_SIM_TIME:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0],
                                     obs['poses_theta'][0], work['tlad'], work['vgain'])
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        sim_time += step_reward
        steps += 1
    real_time = time.time() - start

    return {
        'planner': 'Pure Pursuit',
        'lap_counts': obs['lap_counts'][0],
        'lap_times': obs['lap_times'][0],
        'sim_time': sim_time,
        'real_time': real_time,
        'steps': steps,
        'collided': bool(obs['collisions'][0]),
    }


def benchmark_mpc(conf, control_dt=0.05):
    planner = MPCPlanner(conf, 0.17145 + 0.15875, dt=control_dt)
    env = make_env(conf)

    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

    sim_time, steps = 0.0, 0
    substeps = max(int(round(control_dt / env.timestep)), 1)
    start = time.time()
    while not done and sim_time < MAX_SIM_TIME:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0],
                                     obs['poses_theta'][0], obs['linear_vels_x'][0])
        for _ in range(substeps):
            obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
            sim_time += step_reward
            steps += 1
            if done or sim_time >= MAX_SIM_TIME:
                break
    real_time = time.time() - start

    return {
        'planner': 'MPC',
        'lap_counts': obs['lap_counts'][0],
        'lap_times': obs['lap_times'][0],
        'sim_time': sim_time,
        'real_time': real_time,
        'steps': steps,
        'collided': bool(obs['collisions'][0]),
    }


def print_result(result):
    print('=' * 60)
    print(f"Planner:            {result['planner']}")
    print(f"Laps completed:     {result['lap_counts']:.0f}")
    if result['lap_counts'] > 0:
        print(f"Total sim time:     {result['sim_time']:.3f} s  (over {result['lap_counts']:.0f} laps)")
        print(f"Avg sim time/lap:   {result['sim_time'] / result['lap_counts']:.3f} s")
    else:
        status = 'collided' if result['collided'] else 'timed out'
        print(f"DID NOT FINISH a lap ({status} at sim time {result['sim_time']:.3f} s)")
    print(f"Sim steps:          {result['steps']}")
    print(f"Real (wall) time:   {result['real_time']:.3f} s")
    print(f"Sim/real speedup:   {result['sim_time'] / result['real_time']:.1f}x")
    print('=' * 60)


if __name__ == '__main__':
    conf = load_conf()

    pp_result = benchmark_pure_pursuit(conf)
    print_result(pp_result)

    mpc_result = benchmark_mpc(conf)
    print_result(mpc_result)

    print()
    if pp_result['lap_counts'] > 0:
        pp_lap = pp_result['sim_time'] / pp_result['lap_counts']
        print(f"Pure Pursuit avg lap: {pp_lap:.3f} s")
    if mpc_result['lap_counts'] > 0:
        mpc_lap = mpc_result['sim_time'] / mpc_result['lap_counts']
        print(f"MPC avg lap:          {mpc_lap:.3f} s")
    else:
        print(f"MPC did not complete a lap (DNF at {mpc_result['sim_time']:.3f} s sim time)")
