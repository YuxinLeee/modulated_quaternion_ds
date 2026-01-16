import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import null_space

from src.se3_lpvds.src.util import plot_tools, load_tools, process_tools, quat_tools
from src.se3_lpvds.src.se3_class import se3_class
from src.se3_lpvds.src.quaternion_ds.src.util.plot_tools import plot_traj as plot_traj_ori


def learn_ds(task_name):

    if task_name == "infeasible_task2":
        folder_path = './data/infeasible_task2_0214/all.mat'
        p_raw, q_raw, t_raw, dt = load_tools._process_bag(folder_path, if_flip=False)
        T = t_raw[0][-1] - t_raw[0][0]

    elif task_name == "test_oculus":
        folder_path = './data/test_oculus'
        p_raw, q_raw, t_raw, dt = load_tools.load_single_h5_UMI(folder_path, fixed_time = True)
        T = 4.0

    elif task_name == "clfd_task1":
        T = 5
        p_raw, q_raw, t_raw, dt = load_tools.load_clfd_dataset(task_id=1, num_traj=1, sub_sample=4, duration=T)

    p_in_list, q_in_list, t_in_list   = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
    p_out_list, q_out_list            = process_tools.compute_output(p_in_list, q_in_list, t_in_list)
    p_init, q_init, p_att, q_att      = process_tools.extract_state(p_in_list, q_in_list)
    p_in, q_in, p_out, q_out          = process_tools.rollout_list(p_in_list, q_in_list, p_out_list, q_out_list)
    
    t_in = np.hstack(t_in_list)


    se3_obj = se3_class(p_in, q_in, p_out, q_out, p_att, q_att, dt, K_init=1)
    se3_obj.begin()
    se3_obj.t_in = t_in
    se3_obj.T    = T
    se3_obj.save("models/" + task_name + "_ds.pkl")

    p_test_list = []
    q_test_list = []
    for p_0, q_0 in zip(p_init, q_init):
        q_0 = R.from_quat(q_0.as_quat())
        p_test, q_test, gamma_pos, gamma_ori, v_test, w_test = se3_obj.sim(p_0, q_0, p_att, q_att, step_size=dt, duration=T)
        p_test_list.append(p_test)
    # plot_traj_ori(q_in_list, q_test)


    from src.se3_lpvds.src.lpvds.src.damm.src.util.plot_tools import plot_gmm
    plot_gmm(p_in, se3_obj.pos_ds.damm.z, se3_obj.pos_ds.damm)

    from src.se3_lpvds.src.lpvds.src.util.plot_tools import plot_ds, plot_gamma
    plot_ds(p_in, p_test_list, se3_obj.pos_ds)
    # plot_gamma(gamma_pos, title="pos")
    plot_gamma(gamma_ori, title="ori")

    from src.se3_lpvds.src.util import plot_tools
    plot_tools.plot_result(p_in, p_test, q_test)

    plt.show()



if __name__ == "__main__":
    task_name = "infeasible_task2"
    learn_ds(task_name)