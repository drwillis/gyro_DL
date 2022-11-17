# Hamiltonian-based Neural ODE Networks on the SE(3) Manifold For Dynamics Learning and Control, RSS 2021
# Thai Duong, Nikolay Atanasov

# code structure follows the style of HNN by Greydanus et al. and SymODEM by Zhong et al.
# https://github.com/greydanus/hamiltonian-nn
# https://github.com/Physics-aware-AI/Symplectic-ODENet

import torch, os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from se3hamneuralode import L2_loss, from_pickle, MLP, PSD, SE3HamNODE
from data_gyro import sample_gym

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

gpu=0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

def get_model():
    model =  SE3HamNODE(device=device, pretrain = False, udim=2).to(device)
    path = 'data/gyro-se3ham-rk4-5p.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path = 'data/gyro-se3ham-rk4-5p-stats.pkl'
    stats = from_pickle(path)
    return model, stats

if __name__ == "__main__":
    # Figure and font size
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4

    # t_final = .01
    # dt = 0.001
    # timesteps = int(t_final/dt)
    # times = np.arange(timesteps)*dt
    # # taus = 0.5 * np.square(np.pi * times) - 0.25
    # actions = np.array([[0.0, 0.5],
    #                [-0.25,0.1],
    #                [0.25,0.1]]);
    # trajs, tspan, _ = sample_gym(seed=0, trials=1, timesteps=10000, actions=actions, ori_rep='rotmat', dt=dt)
    # figsize = (10, 7.5)
    # fig = plt.figure(figsize=figsize)
    # ax_px = plt.subplot(421)
    # ax_py = plt.subplot(423)
    # ax_th = plt.subplot(425)
    #
    # ax_vx = plt.subplot(422)
    # ax_vy = plt.subplot(424)
    # ax_thdot = plt.subplot(426)
    #
    # ax_px.plot(tspan, trajs[:, :, 0])
    # ax_px.set_ylabel('x (m)')
    #
    # ax_py.plot(tspan, trajs[:, :, 1])
    # ax_py.set_ylabel('y (m)')
    #
    # ax_th.plot(tspan, trajs[:, :, 2])
    # ax_th.set_ylabel('$\omega$ (rad)')
    #
    # ax_vx.plot(tspan, trajs[:, :, 12])
    # ax_vx.set_ylabel(r'$\frac{dx}{dt}$ (m/sec)')
    #
    # ax_vy.plot(tspan, trajs[:, :, 13])
    # ax_vy.set_ylabel(r'$\frac{dy}{dt}$ (m/sec)')
    #
    # ax_thdot.plot(tspan, trajs[:, :, 17])
    # ax_thdot.set_ylabel(r'$\frac{d\omega}{dt}$ (rad/sec)')
    #
    # ax_px.set_title('Position')
    # ax_vx.set_title('Velocity')
    # ax_thdot.set_xlabel('Time (s)')
    # ax_th.set_xlabel('Time (s)')
    #
    # plt.subplots_adjust(left=0.1, right=0.98, top=0.93, wspace=0.3)
    # #plt.savefig('./png/tracking_results.pdf', bbox_inches='tight', pad_inches=0.1)
    # plt.show()

    # Plot loss
    # Load trained model
    model, stats = get_model()

    # Load train/test data
    train_x_hat = stats['train_x_hat']
    test_x_hat = stats['test_x_hat']
    train_x = stats['train_x']
    test_x = stats['test_x']
    t_eval = stats['t_eval']
    print("Loaded data!")
    fig = plt.figure(figsize=figsize)
    train_loss = stats['train_loss']
    test_loss = stats['test_loss']
    iterations = len(train_loss)
    plt.plot(train_loss, 'b', linewidth=line_width, label='train loss')
    plt.plot(test_loss, 'r--', linewidth=line_width, label='test loss')
    plt.xlabel("iterations", fontsize=fontsize_ticks)
    plt.yscale('log')
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/loss_log.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Pick a sample test trajectory
    traj = 2
    sample_traj = test_x[traj,:,2,0:12]
    sample_traj_hat = test_x_hat[traj,:,2,0:12]

    # Check SE(3) constraints along the trajectory
    det = []
    RRT_I_dist = []
    for i in range(len(sample_traj_hat)):
        R_hat = sample_traj_hat[i,3:12]
        R_hat = R_hat.reshape(3,3)
        R_det = np.linalg.det(R_hat)
        det.append(np.abs(R_det - 1))
        R_RT = np.matmul(R_hat, R_hat.transpose())
        RRT_I = np.linalg.norm(R_RT - np.diag([1.0, 1.0, 1.0]))
        RRT_I_dist.append(RRT_I)
    # Plot SE(3) constraints along the trajectory
    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, det, 'b', linewidth=line_width, label=r'$|det(R) - 1|$')
    plt.plot(t_eval, RRT_I_dist, 'r', linewidth=line_width, label=r'$\Vert R R^\top - I\Vert$')
    plt.xlabel("t", fontsize=fontsize_ticks)
    plt.ylim(0.0, 0.0000005)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/SO3_constraints_test.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # This is the generalized coordinates q = pose
    pose = torch.tensor(sample_traj, requires_grad=True, dtype=torch.float32).to(device)
    x, R = torch.split(pose, [3, 9], dim=1)

    # Calculate the M^-1, V, g for the q.
    M_q_inv1 = model.M_net1(x)
    M_q_inv2 = model.M_net2(R)
    V_q = model.V_net(pose)
    g_q = model.g_net(pose)

    # Plot V(q)
    fig = plt.figure(figsize=figsize)
    temp = pose[:, 2]
    plt.plot(sample_traj[:,2], V_q.detach().cpu().numpy(), 'b--', label=r'$V(q)$', linewidth=3)
    plt.xlabel("$z$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/V_x.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Plot M1^-1(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 2, 2], 'r--', linewidth=line_width,
             label=r'$M^{-1}_{1}(q)[0,0]$')
    plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 0, 0], 'g--', linewidth=line_width,
             label=r'$M^{-1}_{1}(q)[1,1]$')
    plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 1, 1], 'b--', linewidth=line_width,
             label=r'$M^{-1}_{1}(q)[2,2]$')
    plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 0, 1], 'c--', linewidth=line_width,
             label=r'Other $M^{-1}_{1}(q)[i,j]$')
    plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 0, 2], 'c--', linewidth=line_width)
    plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 1, 0], 'c--', linewidth=line_width)

    plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 1, 2], 'c--', linewidth=line_width)
    plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 2, 0], 'c--', linewidth=line_width)
    plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 2, 1], 'c--', linewidth=line_width)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/M1_x_all.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Plot M2^-1(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:,0, 0], 'r--', linewidth=line_width,
             label=r'$M^{-1}_{2}(q)[0, 0]$')
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 1, 1],'g--', linewidth=line_width,
             label=r'$M^{-1}_{2}(q)[1, 1]$')
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:,2, 2], 'b--',linewidth=line_width,
             label=r'$M^{-1}_{2}(q)[2,2]$')
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 0, 1], 'c--', linewidth=line_width,
             label=r'Other $M^{-1}_{2}(q)[i,j]$')
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 0, 2], 'c--', linewidth=line_width)
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 1, 0], 'c--', linewidth=line_width)
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 1, 2], 'c--', linewidth=line_width)
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 2, 0], 'c--', linewidth=line_width)
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 2, 1], 'c--', linewidth=line_width)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize, loc = 'lower right')
    plt.savefig('./png/M2_x_all.png', bbox_inches='tight')
    plt.show()

    # Plot g_v(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 0, 0], 'c--', linewidth=line_width,
             label=r'Other $g_{v}(q)$')
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 1, 0], 'c--', linewidth=line_width)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 2, 0], 'b--', linewidth=line_width,
             label=r'$g_{v}(q)[2,0]$')
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/g_v_x.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Plot g_omega(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 1], 'r--', linewidth=line_width,
             label=r'$g_{\omega}(q)[0,1]$')
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 2], 'g--', linewidth=line_width,
             label=r'$g_{\omega}(q)[1,2]$')
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 3], 'b--', linewidth=line_width,
             label=r'$g_{\omega}(q)[2,3]$')
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 0], 'c--', linewidth=2,
             label=r'Other $g_{\omega}(q)$')
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 2], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 3], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 0], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 1], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 3], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 0], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 1], 'c--', linewidth=2)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 2], 'c--', linewidth=2)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./png/g_w_x.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
