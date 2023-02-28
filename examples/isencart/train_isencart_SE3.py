# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch, argparse
import numpy as np
import pandas as pd
import os, sys
from torchdiffeq import odeint_adjoint as odeint
from se3hamneuralode import MLP, PSD
from se3hamneuralode import SE3HamNODE
from data_collection import get_dataset, arrange_data
from se3hamneuralode import to_pickle, pose_L2_geodesic_loss, traj_pose_L2_geodesic_loss
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

THIS_DIR = os.path.dirname(os.path.abspath(__file__))+'/data'
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learn_rate', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=500, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=10, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='isencart', type=str, help='only one option right now')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=5,
                        help='number of evaluation points by the ODE solver, including the initial point')
    parser.add_argument('--solver', default='rk4', type=str, help='type of ODE Solver for Neural ODE')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total


def get_marker_coords(dataset_row):
    return np.array([[dataset_row[2], dataset_row[0], dataset_row[1]],
                     [dataset_row[5], dataset_row[3], dataset_row[4]],
                     [dataset_row[8], dataset_row[6], dataset_row[7]],
                     [dataset_row[11], dataset_row[9], dataset_row[10]]])/1000
    # return np.array([dataset_row[0:3:1],
    #                  dataset_row[3:6:1],
    #                  dataset_row[6:9:1],
    #                  dataset_row[9:12:1]])

def process_data(data_set, t_eval, reference_coordinate_sys_points = None):
    training_data = {}
    ALIGNMENT_ERROR_TOLERANCE = 0.008 #

    # delete marker ID fields for Marker 4, Marker 3, Marker 2, Marker 1 respectively
    data_set = np.delete(data_set, 13, axis=1)
    data_set = np.delete(data_set, 9, axis=1)
    data_set = np.delete(data_set, 5, axis=1)
    data_set = np.delete(data_set, 1, axis=1)
    data_set = np.delete(data_set, 0, axis=1)

    # data_set = np.delete(data_set, np.s_[2000::], axis=0)

    if reference_coordinate_sys_points is None:
        reference_coordinate_sys_points = get_marker_coords(data_set[0, :])

    training_data['data'] = np.zeros((data_set.shape[0], 20), dtype=np.float64)
    training_data['time'] = np.zeros((data_set.shape[0], 1), dtype=np.float64)
    training_data['input_valid'] = np.zeros((data_set.shape[0], 1), dtype=bool)
    training_data['data_valid'] = np.ones((data_set.shape[0], 1), dtype=bool)
    # CONVERT DATA TO x,R,dx,dR,Vbatt,dutyL,dutyR
    # p_x = np.mean(data_set[:, 0:10:3], axis=1)
    # p_y = np.mean(data_set[:, 1:11:3], axis=1)
    # p_z = np.mean(data_set[:, 2:12:3], axis=1)
    #
    # v_x = np.concatenate([[0], np.diff(p_x)], axis=0)
    # v_y = np.concatenate([[0], np.diff(p_y)], axis=0)
    # v_z = np.concatenate([[0], np.diff(p_z)], axis=0)

    from ralign import ralign, similarity_transform
    R_bodyframe = np.eye(3)
    R_0 = None
    T_prev = np.eye(4)
    R_prev = np.eye(3)
    measured_coordinate_sys_points_cur = get_marker_coords(data_set[0, :]).copy()
    coord_sys_origin = measured_coordinate_sys_points_cur.mean(axis=0)
    position_prev = coord_sys_origin
    dt = t_eval[1] - t_eval[0]
    for row in range(data_set.shape[0]):
        if row > 0:
            dt = t_eval[row] - t_eval[row-1]
        measured_coordinate_sys_points_cur = get_marker_coords(data_set[row, :])
        # R, c, t = ralign(reference_coordinate_sys_points.T, measured_coordinate_sys_points_cur.T)
        R_cur, c, t = similarity_transform(reference_coordinate_sys_points, measured_coordinate_sys_points_cur)
        if R_0 is None:
            R_0 = R_cur
        # print("t = " + str(t-coord_sys_origin))
        if abs(c - 1.0) > 1.0e-1:
            print("Error! Invalid scale! c = " + str(c))
            training_data['data_valid'][row] = False
        T_cur = np.concatenate((np.concatenate([R_cur, t.reshape(3, 1)], axis=1), [[0, 0, 0, 1.0]]), axis=0)
        # if (row==0):
        #     dR, dc, dt = np.eye(3), 1.0, np.zeros(3)
        # else:
        #     measured_coordinate_sys_points_prev = get_marker_coords(data_set[row - 1, :])
        #     dR, dc, dt = ralign(measured_coordinate_sys_points_prev.T, measured_coordinate_sys_points_cur.T)
        T_delta12 = T_cur @ np.linalg.inv(T_prev)
        dR = T_delta12[:3, :3]
        dR = R_cur @ R_prev.T
        transformed_points = np.concatenate([measured_coordinate_sys_points_cur, [[1], [1], [1], [1]]],
                                            axis=1) @ T_delta12.T
        transformed_points2 = np.concatenate([reference_coordinate_sys_points, [[1], [1], [1], [1]]], axis=1) @ T_cur.T
        pt_errors = measured_coordinate_sys_points_cur - transformed_points2[:, :3]
        # if row == 62 or row == 63:
        #     aaa = 1
        if (1 / 4) * np.sum(np.sqrt(np.sum(pt_errors ** 2, axis=1))) > ALIGNMENT_ERROR_TOLERANCE:
            print("Bad data detected at row " + str(row))
            print("average pt error = " + str((1 / 4) * np.sum(np.sqrt(np.sum(pt_errors ** 2, axis=1)))))
            training_data['data_valid'][row] = False
        else:
            training_data['data_valid'][row] = True
        # print("Rotation matrix=\n", R, "\nScaling coefficient=", c, "\nTranslation vector=", t)
        # print("derivative Rotation matrix=\n", dR, "\nScaling coefficient=", dc, "\nTranslation vector=", dt)
        position = measured_coordinate_sys_points_cur.mean(axis=0)
        # print("position="+str(position))
        velocity = (position - position_prev) / dt
        velocity2 = T_delta12[:3, 3].T
        training_data['time'] = t_eval[row]
        # if data_set[row, 14] != 0 or data_set[row, 15] != 0:
        # if abs(data_set[row, 14]) > 0.1 or abs(data_set[row, 15]) > 0.1:
        if data_set[row, 14] > 0.1 and data_set[row, 15] > 0.1:
            training_data['input_valid'][row] = True
        else:
            training_data['input_valid'][row] = False
            # training_data['data_valid'][row] = False
        training_data['data'][row, 0:3] = position
        training_data['data'][row, 3:12] = R_cur.flatten()
        theta = np.arccos((np.trace(R_cur) - 1) / 2.0)
        sin_theta = np.sin(theta)
        if abs(sin_theta) > 1.0e-6:
            theta_over_sin_theta = theta / sin_theta
        else:
            theta_over_sin_theta = 1
        omega_SE3 = np.array((dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1])) / (2.0 * dt)
        if np.linalg.norm(velocity) > 1.0e-3 and np.linalg.norm(position - coord_sys_origin) > 0.5:
            aaa = 1
        # v_bodyframe = R_prev.T @ velocity
        # w_bodyframe = R_prev.T @ omega_SE3
        # v_bodyframe = dR @ np.array([np.linalg.norm(velocity), 0, 0, ])
        # omega_SE3 = omega_SE3 / dt
        # v_bodyframe = dR @ np.array([(1-omega_SE3[2])*np.linalg.norm(velocity), omega_SE3[2]*np.linalg.norm(velocity), 0])
        v_bodyframe = R_cur.T @ velocity
        v_bodyframe[1] = -v_bodyframe[1]
        w_bodyframe = R_cur.T @ omega_SE3
        w_bodyframe[1] = -w_bodyframe[1]
        # w_bodyframe = omega_SE3
        # training_data['data'][row, 12:15] = velocity
        training_data['data'][row, 12:15] = v_bodyframe
        # if abs(velocity[0] - T_delta12[0, 3]) > 1.0e-5:
        #     print("Error! " + str(velocity[0] - T_delta12[0, 3]))
        # training_data['data'][row, 15:18] = omega_SE3
        training_data['data'][row, 15:18] = w_bodyframe
        #  V_batt
        # training_data[row, 19] = data_set[row, 22]
        #  dutyR, dutyL
        # training_data['data'][row, 18] = 0.01*(data_set[row, 14]**2 + data_set[row, 15]**2)
        training_data['data'][row, 18] = data_set[row, 14]
        training_data['data'][row, 19] = data_set[row, 15]
        position_prev = position
        T_prev = T_cur
        R_prev = R_cur
    return training_data, reference_coordinate_sys_points

def process_data2(data_set, t_eval, reference_coordinate_sys_points = None):
    training_data = {}

    training_data['data'] = np.zeros((data_set.shape[0], 20), dtype=np.float64)
    training_data['time'] = np.zeros((data_set.shape[0], 1), dtype=np.float64)
    training_data['input_valid'] = np.zeros((data_set.shape[0], 1), dtype=bool)
    training_data['data_valid'] = np.ones((data_set.shape[0], 1), dtype=bool)
    coord_sys_origin = np.array([0, 0, 0])
    R_prev = np.eye(3)
    position_prev = data_set[0, 0:3]
    dt = t_eval[1] - t_eval[0]
    for row in range(data_set.shape[0]):
        if row > 0:
            dt = t_eval[row] - t_eval[row-1]
        position_cur = data_set[row, 0:3]
        training_data['data'][row, 0:3] = position_cur
        velocity = (position_cur - position_prev) / dt
        quat = data_set[row,3:7]
        if abs(np.linalg.norm(quat)-1) > 1.0e-5:
            aa = 1
        R_cur = Rotation.from_quat(quat).as_matrix()
        dR = R_cur @ R_prev.T
        training_data['data'][row, 3:12] = R_cur.flatten()
        theta = np.arccos((np.trace(R_cur) - 1) / 2.0)
        sin_theta = np.sin(theta)
        if abs(sin_theta) > 1.0e-6:
            theta_over_sin_theta = theta / sin_theta
        else:
            theta_over_sin_theta = 1
        # omega_SE3 = np.array((dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1])) / 2.0
        omega_SE3 = data_set[row, 7:10]
        if np.linalg.norm(velocity) > 1.0e-3 and np.linalg.norm(position_prev - coord_sys_origin) > 0.5:
            aaa = 1
        # v_bodyframe = dR @ np.array([np.linalg.norm(velocity), 0, 0, ])
        v_bodyframe = R_cur.T @ velocity
        w_bodyframe = R_cur.T @ omega_SE3
        # training_data['data'][row, 12:15] = velocity
        training_data['data'][row, 12:15] = v_bodyframe
        # if abs(velocity[0] - T_delta12[0, 3]) > 1.0e-5:
        #     print("Error! " + str(velocity[0] - T_delta12[0, 3]))
        # training_data['data'][row, 15:18] = omega_SE3
        training_data['data'][row, 15:18] = w_bodyframe
        #  dutyR, dutyL
        training_data['data'][row, 18] = data_set[row, 13] + data_set[row, 14]
        training_data['data'][row, 19] = data_set[row, 13] - data_set[row, 14]
        R_prev = R_cur
        position_prev = position_cur

    return training_data, reference_coordinate_sys_points


def train(args):

    # Load saved params if needed
    #path = '{}/quadrotor-se3ham-rk4-5p2-2000.tar'.format(args.save_dir)
    #model.load_state_dict(torch.load(path, map_location=device))
    # Save/load pre-init model
    #path = '{}/quadrotor-se3ham-rk4-5p-pre-init.tar'.format(args.save_dir)
    #model.load_state_dict(torch.load(path, map_location=device))
    #torch.save(model.state_dict(), path)

    # Collect data
    # data = get_dataset(test_split=0.8, save_dir=args.save_dir)
    USE_RAW_DATASET = False

    if USE_RAW_DATASET:
        # data_file_list = ['data/data_1.txt']
        # data_file_list = ['data/data_2.txt']
        data_file_list = ['data/data_3.txt']
        # data_file_list = ['data/data_1.txt', 'data/data_2.txt', 'data/data_3.txt', 'data/data_4.txt']
        reference_coordinate_sys_points = None

        for data_file in data_file_list:
            data_set = pd.read_csv(data_file).to_numpy()
            # delete first row of zeros
            data_set = np.delete(data_set, 0, axis=0)
            t_eval = data_set[:, 0].copy()
            t_eval = t_eval - t_eval[0]
            training_data, reference_coordinate_sys_points = process_data(data_set, t_eval, reference_coordinate_sys_points)
    else:
        data_file_list = ['data/fixed_data2_3.txt']
        reference_coordinate_sys_points = None
        for data_file in data_file_list:
            data_set = pd.read_csv(data_file).to_numpy()
            # delete first row of zeros
            data_set = np.delete(data_set, 0, axis=0)
            t_eval = data_set[:, 0]
            t_eval = t_eval - t_eval[0]
            data_set = np.delete(data_set, 0, axis=1)
            training_data, reference_coordinate_sys_points = process_data2(data_set, t_eval, reference_coordinate_sys_points)

    scalef = 10
    subsample = 10
    plt.figure()
    for endIdx in range(50,training_data['data'].shape[0], 1000):
        plt.plot(training_data['data'][:endIdx, 0],training_data['data'][:endIdx, 1])
        plt.quiver(training_data['data'][::subsample, 0], training_data['data'][::subsample, 1],
                   # scalef*training_data['data'][::subsample, 3], scalef*training_data['data'][::subsample, 6])
                   scalef * training_data['data'][::subsample, 12], scalef * training_data['data'][::subsample, 13])
        plt.quiver(training_data['data'][::subsample, 0], training_data['data'][::subsample, 1],
                   -training_data['data'][::subsample, 17]*training_data['data'][::subsample, 13],
                   training_data['data'][::subsample, 17]*training_data['data'][::subsample, 12], color='r')
        #plt.show()
        plt.axis('equal')
        plt.draw()
        plt.pause(0.15)

    from scipy.signal import savgol_filter
    training_data_smooth = { 'data': {}}
    training_data_smooth['data'] = training_data['data'].copy()
    smoothing_channels = (0, 1, 2, 12, 13, 14, 15, 16, 17, 18, 19)
    for smoothing_channel_idx in smoothing_channels:
        # training_data_smooth['data'][:, smoothing_channel_idx] = savgol_filter(training_data['data'][:, smoothing_channel_idx], 7, 3)
        training_data_smooth['data'][:, smoothing_channel_idx] = savgol_filter(training_data['data'][:, smoothing_channel_idx], 51, 3)

    # Create two subplots and unpack the output array immediately
    idxs = range(0,len(training_data))

    fv, subplot_axes_v = plt.subplots(3,1)
    subplot_axes_v[0].plot(training_data['data'][:, 12], 'r', label='v_x')
    subplot_axes_v[1].plot(training_data['data'][:, 13], 'r', label='v_y')
    subplot_axes_v[2].plot(training_data['data'][:, 14], 'r', label='v_z')
    subplot_axes_v[0].plot(training_data_smooth['data'][:, 12], 'b', label='smoothed v_x')
    subplot_axes_v[1].plot(training_data_smooth['data'][:, 13], 'b', label='smoothed v_y')
    subplot_axes_v[2].plot(training_data_smooth['data'][:, 14], 'b', label='smoothed v_z')
    plt.draw()
    plt.pause(0.001)

    f_w, subplot_axes = plt.subplots(3,1)
    subplot_axes[0].plot(training_data['data'][:, 15], 'r', label='dw_x')
    subplot_axes[1].plot(training_data['data'][:, 16], 'r', label='dw_y')
    subplot_axes[2].plot(training_data['data'][:, 17], 'r', label='dw_z')
    subplot_axes[0].plot(training_data_smooth['data'][:, 15], 'b', label='smoothed dw_x')
    subplot_axes[1].plot(training_data_smooth['data'][:, 16], 'b', label='smoothed dw_y')
    subplot_axes[2].plot(training_data_smooth['data'][:, 17], 'b', label='smoothed dw_z')
    plt.draw()
    plt.pause(0.001)

    f_input, subplot_axes_input = plt.subplots(2,1)
    subplot_axes_input[0].plot(training_data['data'][:, 18], 'r', label='inputL')
    subplot_axes_input[1].plot(training_data['data'][:, 19], 'r', label='inputR')
    subplot_axes_input[0].plot(training_data_smooth['data'][:, 18], 'b', label='smoothed inputL')
    subplot_axes_input[1].plot(training_data_smooth['data'][:, 19], 'b', label='smoothed inputR')
    plt.draw()
    plt.pause(0.001)

    non_zero_input_indices = np.nonzero(training_data['input_valid'])
    EXPERIMENT_NUMSAMPLES = 5
    num_samples = training_data['data'].shape[0]
    trim = num_samples % EXPERIMENT_NUMSAMPLES
    num_experiments = (num_samples-trim)/EXPERIMENT_NUMSAMPLES
    num_experiments = int(num_experiments)
    training_data['data'] = training_data['data'][trim:]
    deleteIndices = []
    for experimentIdx in range(0, num_experiments):
        if not np.all(training_data['data_valid'][experimentIdx*EXPERIMENT_NUMSAMPLES:((experimentIdx+1)*EXPERIMENT_NUMSAMPLES)]):
            #experiments[:, experimentIdx, :] = training_data_smooth['data'][experimentIdx*EXPERIMENT_NUMSAMPLES:((experimentIdx+1)*EXPERIMENT_NUMSAMPLES),:]
            #experiments.(training_data_smooth['data'][experimentIdx*EXPERIMENT_NUMSAMPLES:((experimentIdx+1)*EXPERIMENT_NUMSAMPLES),:])
            # else:
            print("Deleting experiment " + str(experimentIdx))
            deleteIndices.append(experimentIdx)

    MAX_EXPERIMENTS = 50000
    num_available_experiments = num_experiments-len(deleteIndices)
    num_experiments = min(num_available_experiments, MAX_EXPERIMENTS)

    experiments = np.zeros((EXPERIMENT_NUMSAMPLES, num_experiments, 20))
    filteredIdx = 0
    for experimentIdx in range(0, num_available_experiments):
        if deleteIndices.count(experimentIdx) == 0:
            experiments[:, filteredIdx, :] = training_data['data'][experimentIdx*EXPERIMENT_NUMSAMPLES:((experimentIdx+1)*EXPERIMENT_NUMSAMPLES),:]
            # experiments[:, filteredIdx, :] = training_data_smooth['data'][experimentIdx*EXPERIMENT_NUMSAMPLES:((experimentIdx+1)*EXPERIMENT_NUMSAMPLES),:]
            filteredIdx += 1
            if filteredIdx == MAX_EXPERIMENTS:
                break


    delta_t = np.mean(np.diff(t_eval))
    experiment_times = np.linspace(0,(EXPERIMENT_NUMSAMPLES-1)*delta_t, EXPERIMENT_NUMSAMPLES)

    pct_train_samples_split = 0.5
    # samples = training_data['data'].shape[1]
    samples = experiments.shape[1]
    split_ix = max(1, int(samples * pct_train_samples_split))
    # split_ix = int(EXPERIMENT_NUMSAMPLES * test_split)
    split_data = {}

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the model
    if args.verbose:
        print("Start training with num of points = {} and solver {}.".format(args.num_points, args.solver))
    model = SE3HamNODE(device=device, pretrain=True, udim=2).to(device)

    num_parm = get_model_parm_nums(model)
    print('Model contains {} parameters'.format(num_parm))
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0.0)

    # split_data['x'], split_data['test_x'] = training_data['data'][:split_ix, :], training_data['data'][split_ix:, :]
    # split_data['t_train'], split_data['t_test'] = t_eval[:split_ix], t_eval[split_ix:]
    split_data['x'], split_data['test_x'] = experiments[:, :split_ix, :], experiments[:, split_ix:, :]
    # split_data['test_x'], split_data['x'] = experiments[:, :split_ix, :], experiments[:, split_ix:, :]
    split_data['t_train'], split_data['t_test'] = experiment_times[:split_ix], experiment_times[split_ix:]
    data = split_data
    # data['t'] = tspan
    # df.head()  # To get first n rows from the dataset default value of n is 5
    # M = len(df)

    train_x_cat = data['x']
    test_x_cat = data['test_x']

    train_x_cat = torch.tensor(train_x_cat, requires_grad=True, dtype=torch.float32).to(device)
    test_x_cat = torch.tensor(test_x_cat, requires_grad=True, dtype=torch.float32).to(device)
    #train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32).to(device)
    #test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32).to(device)
    t_eval_host = np.linspace(0, (EXPERIMENT_NUMSAMPLES - 1) * delta_t, 5)
    # t_eval = torch.tensor(split_data['t_train'], requires_grad=True, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_eval_host, requires_grad=True, dtype=torch.float32).to(device)
    data['t'] = t_eval_host


    # Training stats
    stats = {'train_loss': [], 'test_loss': [], 'forward_time': [], 'backward_time': [], 'nfe': [], 'train_x_loss': [],\
             'test_x_loss':[], 'train_v_loss': [], 'test_v_loss': [], 'train_w_loss': [], 'test_w_loss': [], 'train_geo_loss':[], 'test_geo_loss':[]}
    # Start training
    plt.figure()
    for step in range(0,args.total_steps + 1):
        #print(step)
        train_loss = 0
        test_loss = 0
        train_x_loss = 0
        train_v_loss = 0
        train_w_loss = 0
        train_geo_loss = 0
        test_x_loss = 0
        test_v_loss = 0
        test_w_loss = 0
        test_geo_loss = 0

        t = time.time()
        # Predict states
        #  returns predictions for times in t_eval for each of the provided state values in train_x_cat as initial conditions
        train_x_hat = odeint(model, train_x_cat[0, :, :], t_eval, method=args.solver)
        forward_time = time.time() - t
        target = train_x_cat[1:, :, :]
        target_hat = train_x_hat[1:, :, :]
        # target = train_x_cat
        # target_hat = train_x_hat

        # Calculate loss
        train_loss_mini, x_loss_mini, v_loss_mini, w_loss_mini, geo_loss_mini = \
            pose_L2_geodesic_loss(target, target_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        train_loss = train_loss + train_loss_mini
        train_x_loss = train_x_loss + x_loss_mini
        train_v_loss = train_v_loss + v_loss_mini
        train_w_loss = train_w_loss + w_loss_mini
        train_geo_loss = train_geo_loss + geo_loss_mini

        # Calculate loss for test data
        test_x_hat = odeint(model, test_x_cat[0, :, :], t_eval, method=args.solver)
        target = test_x_cat[1:, :, :]
        target_hat = test_x_hat[1:, :, :]
        test_loss_mini, x_loss_mini, v_loss_mini, w_loss_mini, geo_loss_mini = \
            pose_L2_geodesic_loss(target, target_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        test_loss = test_loss + test_loss_mini
        test_x_loss = test_x_loss + x_loss_mini
        test_v_loss = test_v_loss + v_loss_mini
        test_w_loss = test_w_loss + w_loss_mini
        test_geo_loss = test_geo_loss + geo_loss_mini

        # Gradient descent
        t = time.time()
        if step > 0:
            train_loss_mini.backward()
            optim.step()
            optim.zero_grad()
        backward_time = time.time() - t

        # Logging stats
        stats['train_loss'].append(train_loss.item())
        stats['test_loss'].append(test_loss.item())
        stats['train_x_loss'].append(train_x_loss.item())
        stats['test_x_loss'].append(test_x_loss.item())
        stats['train_v_loss'].append(train_v_loss.item())
        stats['test_v_loss'].append(test_v_loss.item())
        stats['train_w_loss'].append(train_w_loss.item())
        stats['test_w_loss'].append(test_w_loss.item())
        stats['train_geo_loss'].append(train_geo_loss.item())
        stats['test_geo_loss'].append(test_geo_loss.item())
        stats['forward_time'].append(forward_time)
        stats['backward_time'].append(backward_time)
        stats['nfe'].append(model.nfe)

        if step % (args.print_every/2) == 0 and len(stats['train_loss']) > 10:
            line_width = 4
            plt.plot(stats['train_loss'], 'b', linewidth=line_width, label='train loss')
            plt.plot(stats['test_loss'], 'r--', linewidth=line_width, label='test loss')
            plt.draw()
            plt.pause(0.001)

        if step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, train_loss.item(), test_loss.item()))
            print("step {}, train_x_loss {:.4e}, test_x_loss {:.4e}".format(step, train_x_loss.item(),
                                                                            test_x_loss.item()))
            print("step {}, train_v_loss {:.4e}, test_v_loss {:.4e}".format(step, train_v_loss.item(),
                                                                            test_v_loss.item()))
            print("step {}, train_w_loss {:.4e}, test_w_loss {:.4e}".format(step, train_w_loss.item(),
                                                                            test_w_loss.item()))
            print("step {}, train_geo_loss {:.4e}, test_geo_loss {:.4e}".format(step, train_geo_loss.item(),
                                                                                test_geo_loss.item()))
            print("step {}, nfe {:.4e}".format(step, model.nfe))
            # # Uncomment this to save model every args.print_every steps
            # os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
            # label = '-se3ham'
            # path = '{}/{}{}-{}-{}p3-{}.tar'.format(args.save_dir, args.name, label, args.solver, args.num_points, step)
            # torch.save(model.state_dict(), path)

    # Calculate loss mean and standard deviation
    train_x, t_eval = data['x'], data['t']
    test_x, t_eval = data['test_x'], data['t']

    train_x = np.expand_dims(train_x, axis=0)
    test_x = np.expand_dims(test_x, axis=0)

    train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32).to(device)
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)

    train_loss = []
    test_loss = []
    train_l2_loss = []
    test_l2_loss = []
    train_geo_loss = []
    test_geo_loss = []
    train_data_hat = []
    test_data_hat = []
    for i in range(train_x.shape[0]):
        train_x_hat = odeint(model, train_x[i, 0, :, :], t_eval, method=args.solver)
        total_loss, l2_loss, geo_loss = \
            traj_pose_L2_geodesic_loss(train_x[i, :, :, :], train_x_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        train_loss.append(total_loss)
        train_l2_loss.append(l2_loss)
        train_geo_loss.append(geo_loss)
        train_data_hat.append(train_x_hat.detach().cpu().numpy())

        # Run test data
        test_x_hat = odeint(model, test_x[i, 0, :, :], t_eval, method=args.solver)
        total_loss, l2_loss, geo_loss = \
            traj_pose_L2_geodesic_loss(test_x[i,:,:,:], test_x_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        test_loss.append(total_loss)
        test_l2_loss.append(l2_loss)
        test_geo_loss.append(geo_loss)
        test_data_hat.append(test_x_hat.detach().cpu().numpy())

    train_loss = torch.cat(train_loss, dim=1)
    train_loss_per_traj = torch.sum(train_loss, dim=0)

    test_loss = torch.cat(test_loss, dim=1)
    test_loss_per_traj = torch.sum(test_loss, dim=0)

    train_l2_loss = torch.cat(train_l2_loss, dim=1)
    train_l2_loss_per_traj = torch.sum(train_l2_loss, dim=0)

    test_l2_loss = torch.cat(test_l2_loss, dim=1)
    test_l2_loss_per_traj = torch.sum(test_l2_loss, dim=0)

    train_geo_loss = torch.cat(train_geo_loss, dim=1)
    train_geo_loss_per_traj = torch.sum(train_geo_loss, dim=0)

    test_geo_loss = torch.cat(test_geo_loss, dim=1)
    test_geo_loss_per_traj = torch.sum(test_geo_loss, dim=0)

    print('Final trajectory train loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
    .format(train_loss_per_traj.mean().item(), train_loss_per_traj.std().item(),
            test_loss_per_traj.mean().item(), test_loss_per_traj.std().item()))
    print('Final trajectory train l2 loss {:.4e} +/- {:.4e}\nFinal trajectory test l2 loss {:.4e} +/- {:.4e}'
    .format(train_l2_loss_per_traj.mean().item(), train_l2_loss_per_traj.std().item(),
            test_l2_loss_per_traj.mean().item(), test_l2_loss_per_traj.std().item()))
    print('Final trajectory train geo loss {:.4e} +/- {:.4e}\nFinal trajectory test loss {:.4e} +/- {:.4e}'
    .format(train_geo_loss_per_traj.mean().item(), train_geo_loss_per_traj.std().item(),
            test_geo_loss_per_traj.mean().item(), test_geo_loss_per_traj.std().item()))

    stats['traj_train_loss'] = train_loss_per_traj.detach().cpu().numpy()
    stats['traj_test_loss'] = test_loss_per_traj.detach().cpu().numpy()
    stats['train_x'] = train_x.detach().cpu().numpy()
    stats['test_x'] = test_x.detach().cpu().numpy()
    stats['train_x_hat'] = np.array(train_data_hat)
    stats['test_x_hat'] = np.array(test_data_hat)
    stats['t_eval'] = t_eval.detach().cpu().numpy()
    return model, stats


if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # Save model
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-se3ham'
    path = '{}/{}{}-{}-{}p.tar'.format(args.save_dir, args.name, label, args.solver, args.num_points)
    torch.save(model.state_dict(), path)
    path = '{}/{}{}-{}-{}p-stats.pkl'.format(args.save_dir, args.name, label, args.solver, args.num_points)
    print("Saved file: ", path)
    to_pickle(stats, path)
