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

from waypoint_follow import PurePursuitPlanner


def benchmark_pure_pursuit(config_path='config_example_map.yaml'):
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312,
            'tlad': 0.82461887897713965, 'vgain': 1.375}

    with open(config_path) as f:
        conf = Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    planner = PurePursuitPlanner(conf, 0.17145 + 0.15875)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                    num_agents=1, timestep=0.01, integrator=Integrator.RK4)

    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

    sim_time = 0.0
    steps = 0
    start = time.time()

    while not done:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0],
                                     obs['poses_theta'][0], work['tlad'], work['vgain'])
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        sim_time += step_reward
        steps += 1

    real_time = time.time() - start

    return {
        'planner': 'PurePursuitPlanner (waypoint_follow.py)',
        'lap_counts': obs['lap_counts'][0],
        'lap_times': obs['lap_times'][0],
        'sim_time': sim_time,
        'real_time': real_time,
        'steps': steps,
    }


if __name__ == '__main__':
    result = benchmark_pure_pursuit()
    print('=' * 60)
    print(f"Planner:            {result['planner']}")
    print(f"Laps completed:     {result['lap_counts']:.0f}")
    print(f"Env lap_times:      {result['lap_times']:.3f} s")
    print(f"Total sim time:     {result['sim_time']:.3f} s  (over {result['lap_counts']:.0f} laps)")
    print(f"Avg sim time/lap:   {result['sim_time'] / max(result['lap_counts'], 1):.3f} s")
    print(f"Sim steps:          {result['steps']}")
    print(f"Real (wall) time:   {result['real_time']:.3f} s")
    print(f"Sim/real speedup:   {result['sim_time'] / result['real_time']:.1f}x")
    print('=' * 60)
