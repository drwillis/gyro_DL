# code structure follows the style of Symplectic ODE-Net
# https://github.com/d-biswa/Symplectic-ODENet/blob/master/experiment-single-embed/data.py

import numpy as np
from se3hamneuralode import to_pickle, from_pickle
import gym
import envs


def sample_gym(seed=0, timesteps=10, trials=1, min_angle=0.,
               verbose=False, actions=np.array([[0.0, 0.0]], dtype=float),
               env_name='MyGyro-v1', ori_rep='rotmat', dt=0.05, friction=False, render=False):
    gym_settings = locals()
    if verbose:
        print("Making a dataset of mems gyroscope observations.")
    env = gym.make(env_name)
    env.seed(seed)
    env.friction = friction
    b = np.array([0, 0, 0, 0, 0, 0]).T
    # tau=0.5*square(2*np.pi*t*(1/2))-.25  % torque applied to gyroscope.  Modify this signal to obtain different data sets

    trajs = []
    for actionIdx in range(actions.shape[0]):
        y0 = actions[actionIdx][1]
        for trialIdx in range(trials):
            valid = False
            while not valid:
                env.reset(ori_rep=ori_rep)
                traj = []
                for step in range(timesteps):
                    if render:
                        env.render()
                    # omega_t = np.random.uniform(low=0, high=2 * np.pi)
                    omega_t = 500*step*dt
                    actions[actionIdx][1] = y0 * np.cos(omega_t)
                    obs, _, _, _ = env.step(actions[actionIdx])  # action
                    x = np.concatenate((obs, actions[actionIdx]))
                    traj.append(x)
                traj = np.stack(traj)
                if np.amax(traj[:, 2]) < env.max_speed - 0.001 and np.amin(traj[:, 2]) > -env.max_speed + 0.001:
                    valid = True
            trajs.append(traj)
    trajs = np.stack(trajs)  # (trials, timesteps, 2)
    trajs = np.transpose(trajs, (1, 0, 2))  # (timesteps, trails, 2)
    tspan = np.arange(timesteps) * dt
    return trajs, tspan, gym_settings


def get_dataset(seed=0, trials=10, test_split=0.5, save_dir=None, actions=np.array([[0, 0]], dtype=float), dt=0.05,
                rad=False, ori_rep='rotmat', friction=False, **kwargs):
    data = {}

    assert save_dir is not None
    path = '{}/gyro-gym-dataset.pkl'.format(save_dir)
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        trajs_force = []
        for actionIdx in range(actions.shape[0]):
            action = actions[actionIdx]
            trajs, tspan, _ = sample_gym(seed=seed, trials=trials, actions=np.array([action]), ori_rep=ori_rep,
                                         friction=friction, dt=dt, **kwargs)
            trajs_force.append(trajs)
        data['x'] = np.stack(trajs_force, axis=0)  # (3, 45, 50, 3)
        # make a train/test split
        split_ix = int(trials * test_split)
        split_data = {}
        split_data['x'], split_data['test_x'] = data['x'][:, :, :split_ix, :], data['x'][:, :, split_ix:, :]

        data = split_data
        data['t'] = tspan

        to_pickle(data, path)
    return data


def arrange_data(x, t, num_points=2):
    '''Arrange data to feed into neural ODE in small chunks'''
    assert num_points >= 2 and num_points <= len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points - 1:
            x_stack.append(x[:, i:-num_points + i + 1, :, :])
        else:
            x_stack.append(x[:, i:, :, :])
    x_stack = np.stack(x_stack, axis=1)
    x_stack = np.reshape(x_stack,
                         (x.shape[0], num_points, -1, x.shape[3]))
    t_eval = t[0:num_points]
    return x_stack, t_eval


if __name__ == "__main__":
    # us = [0.0, -1.0, 1.0, -2.0, 2.0]
    actions = np.array([[0.0, 0.5],
                        [-0.25, 0.1],
                        [0.25, 0.1]]);
    # trajs, tspan, _  = sample_gym(seed=0, trials=50, u=us[0], timesteps=20, ori_rep='6d')
    # t_final = 2.5;
    t_final = 0.1
    dt = 0.0001
    timesteps = int(t_final / dt)
    # times = np.arange(timesteps)*dt
    # taus = 0.5 * np.square(np.pi * times) - 0.25
    # us = taus

    data = get_dataset(seed=0, timesteps=timesteps, save_dir='data', actions=actions,
                       ori_rep='rotmat', trials=1, dt=dt)
    # trajs, tspan, _  = sample_gym(seed=0, trials=50, u=us[0], timesteps = timesteps, dt=dt, ori_rep='angle', render = False)
    print("Done!")
