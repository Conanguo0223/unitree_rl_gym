import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    # v = np.array([ 0.  ,  0.  , -1.0])
    # q_w = quaternion[0]
    # q_vec = quaternion[1:4]
    # a = v * (2.0 * q_w ** 2 - 1.0)
    # b = np.cross(q_vec, v) * q_w * 2.0
    # c = q_vec * np.matmul(q_vec, v)* 2.0
    # return a - b + c
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse
    # stand_up_joint_pos = np.array([
    # 0.00571868, 0.608813, -1.21763, 
    # -0.00571868, 0.608813, -1.21763,
    # 0.00571868, 0.608813, -1.21763, 
    # -0.00571868, 0.608813, -1.21763
    # ])
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    dim_motor_sensor_ = 3 * 12
    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        steps = 0
        step_policy_start = 5000
        # stand_up_joint_pos = np.array([
        #     0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
        #     0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
        # ],
        #                             dtype=float)
        # stand_up_joint_pos = np.array([
        #     0.0162, 0.59, -1.33, -0.0179, 0.593, -1.32,
        #     0.0077, 0.591, -1.35, -0.0119, 0.59, -1.35
        # ],
        #                             dtype=float)
        stand_up_joint_pos=np.array([0.1, 0.8, -1.5, 
                 -0.1, 0.8,-1.5,
                  0.1, 1.0,-1.5,
                  -0.1, 1.0,-1.5],dtype=float)

        stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
            1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ],dtype=float)

    #     stand_down_joint_pos = np.array([ 0.13819177,  1.22198705, -2.7213251 , -0.13822787,  1.22202895,
    #    -2.72135751,  0.47924419,  1.6843266 , -2.72463541, -0.47948911,
    #     1.68462687, -2.72452872], dtype=float)
        standing = False
        dt = 0.002
        runing_time = 0.0
        kds = np.ones_like(kds) * 0.2
        kps = np.ones_like(kps) * 15.0
        # action = np.zeros_like(kds)
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            tau = pd_control(target_dof_pos, d.sensordata[:12], kps, np.zeros_like(kds), d.sensordata[12:12+12], kds)
            if not standing:
                if counter < 250:
                    tau = np.zeros_like(kds)
                else:
                    runing_time += dt
                    if (runing_time < 30):
                        phase = np.tanh((runing_time) / 1.2)
                        # target_dof_pos =stand_down_joint_pos
                        target_dof_pos = stand_down_joint_pos * (1 - phase) + stand_up_joint_pos * phase
                        # target_dof_pos -= default_angles
                        # target_dof_pos = np.zeros_like(kds)
                        kps_test = np.ones_like(kds)*(phase * 50.0 + (1 - phase) * 20.0)
                        kds_test = np.ones_like(kds)*(3.5)
                        # tau = pd_control(target_dof_pos, d.qpos[7:], kps_test, np.zeros_like(kds), d.qvel[6:], kds_test)
                        tau = pd_control(target_dof_pos, d.sensordata[:12], kps_test, np.zeros_like(kds), d.sensordata[12:12+12], kds_test)
                        # tau = tau_test
                        if phase > 0.97:
                            print("finished standing")
                            standing = True
            
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            
            counter += 1
            if counter % control_decimation == 0 and standing:
                # Apply control signal here.
                print("policy go")
                # create observation
                qj = d.sensordata[:12]
                dqj = d.sensordata[12:24]
                quat = d.sensordata[36:40]
                # omega = d.sensordata[40:43]
                # qj = d.qpos[7:]
                # dqj = d.qvel[6:]
                # quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                gravity_orientation = get_gravity_orientation(quat)
                print(gravity_orientation)

                ''' self.base_lin_vel * self.obs_scales.lin_vel, #3
                    self.base_ang_vel  * self.obs_scales.ang_vel, #3
                    self.projected_gravity, #3
                    self.commands[:, :3] * self.commands_scale, #3
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #12
                    self.dof_vel * self.obs_scales.dof_vel, #12
                    self.actions #12
                '''
                obs[:3] = d.qvel[0:3] * lin_vel_scale  # base linear velocity
                # print(d.qvel[0:3])
                obs[3:6] = omega * ang_vel_scale # base angular velocity
                obs[6:9] = gravity_orientation # self.projected_gravity
                obs[9:12] = cmd * cmd_scale # self.commands[:, :3]
                obs[12 : 12 + num_actions] = (qj-stand_up_joint_pos) * dof_pos_scale # dof_pos - default_dof_pos
                obs[12 + num_actions : 12 + 2 * num_actions] = dqj * dof_vel_scale # self.dof_vel
                obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action # self.actions
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + stand_up_joint_pos
                # print(tau)
                
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
