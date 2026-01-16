import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import null_space

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quaternion_ds.src.quat_class import quat_class
from quaternion_ds.src.util import plot_tools, load_tools, process_tools
from quaternion_ds.src.util.quat_tools import riem_log, parallel_transport


def map(att_vec, q):
    """From QUATERNION to 3D"""
    if isinstance(q, R):
        q1 = quat_tools.riem_log(att_vec, q.as_quat())[0, :] #2d array to 1d
    elif isinstance(q, np.ndarray):
        q1 = quat_tools.riem_log(att_vec, q)[0, :] #2d array to 1d
    elif isinstance(q, list):
        q1 = quat_tools.riem_log(R.from_quat(att_vec), q).T

    att_basis = null_space(att_vec.reshape(1, -1))
    v1, _, _, _ = np.linalg.lstsq(att_basis, q1, rcond=None)

    return v1



# def inv_map(traj):
#     """from 3d traj [M, N] to quat: M is number of points"""
#     M = traj.shape[0]
#     new_ori = [R.identity()] * M
#     for i in range(M):
#         ori_i_red = traj[i, :]
#         ori_i = self.normal_basis @ ori_i_red + self.normal_vec
#         ori_i_quat = quat_tools.riem_exp(self.att, ori_i.reshape(1, -1))
#         new_ori[i] = R.from_quat(ori_i_quat[0])

#     return new_ori




if __name__ == "__main__":
    T = 5
    p_raw, q_raw, t_raw, dt = load_tools.load_clfd_dataset(task_id=2, num_traj=2, sub_sample=4, duration=T)
    T = t_raw[0][-1] - t_raw[0][0]

    p_in_list, q_in_list, t_in_list   = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
    p_out_list, q_out_list            = process_tools.compute_output(p_in_list, q_in_list, t_in_list)
    p_init, q_init_list, p_att, q_att = process_tools.extract_state(p_in_list, q_in_list)
    p_in, q_in, p_out, q_out          = process_tools.rollout_list(p_in_list, q_in_list, p_out_list, q_out_list)

    quat_obj = quat_class(q_in, q_out, q_att, dt, K_init=4)
    quat_obj.begin()

    q_in_att = quat_obj.gmm.q_in_att # orientation projected on the tangent plane of attractor
    print(q_in_att.shape)


    """Construct Modulation Data"""
    index = [1, 10, 15]

    q_in_modulation  = [q_in[i] for i in index]
    q_out_original   = []
    q_out_modulation = []

    for i in range(len(index)):
        q_next, gamma, omega = quat_obj._step(q_in_modulation[i], dt)
        q_out_original.append(q_next)
        q_out_modulation.append(q_next * R.from_euler("xyz", [np.pi/4, 0, 0]))
        # print("Original desired state",q_out_original[-1].as_quat())
        # print("Modulated desired state",q_out_modulation[-1].as_quat())


    """Project onto the tangent plane of attractor"""

    q_in_modulation_att  = riem_log(q_att, q_in_modulation)
    q_out_modulation_att = riem_log(q_att, q_out_modulation)
    q_out_original_att   = riem_log(q_att, q_out_original)


    """Verify the projection"""
    att_basis = null_space(q_att.as_quat().reshape(1, -1))
    
    
    print(att_basis)



    # for i in range(len(index)):
    #     q_out_original_att.append()
    #     q_out_modulation_att.append()





    # print(q_in_modulation)

    # q_in_att_list = [q_att] + q_in_att

    # X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    # y = np.squeeze(X * np.sin(X))
    # kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    # gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    # gaussian_process.fit(X, y)






