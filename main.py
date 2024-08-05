import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

# Cartesian impedance control gains.
impedance_pos = np.asarray([100.0, 100.0, 50.0])  # [N/m]
impedance_ori = np.asarray([150.0, 150.0, 150.0])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

# Kp_null = np.array([75.0, 75.0, 50.0, 50.0, 40.0, 25.0])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 1.0

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 1

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 1.0

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002


def xmat2quat(data, site_id):
    target_pose_quat = np.empty(4)
    target_pose_mat = data.site_xmat[site_id].reshape(9, 1).copy()
    mujoco.mju_mat2Quat(target_pose_quat, target_pose_mat)
    return target_pose_quat.copy()


def main() -> None:
    np.set_printoptions(precision=5, suppress=True, linewidth=100)
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("assets/scene.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)

    model.opt.timestep = dt

    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

    joint_names = [
        "kuka_joint1",
        "kuka_joint2",
        "kuka_joint3",
        "kuka_joint4",
        "kuka_joint5",
        "kuka_joint6",
        "kuka_joint7",
    ]
    actuator_names = [
        "actuator1",
        "actuator2",
        "actuator3",
        "actuator4",
        "actuator5",
        "actuator6",
        "actuator7",
    ]

    # joint_names = [
    #         "shoulder_pan_joint",
    #         "upperarm_joint",
    #         "elbow_joint",
    #         "forearm_joint",
    #         "wrist_1_joint",
    #         "wrist_2_joint",
    # ]

    # actuator_names = [
    #         "shoulder_pan_joint",
    #         "upperarm_joint",
    #         "elbow_joint",
    #         "forearm_joint",
    #         "wrist_1_joint",
    #         "wrist_2_joint",
    # ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    # q0 = np.array([0, 0.2082, -1.6390, 0, 1.2944, 0])
    q0 = np.array([-0.002, 0.753, 0.002, -1.313, 0.003, 1.067, 0.0])
    for i in range(len(joint_names)):
        data.joint(joint_names[i]).qpos = q0[i]

    mujoco.mj_forward(model, data)

    ee_site_name = "ee"
    # ee_site_name = "pinch_site"
    ee_site_id = model.site(ee_site_name).id
    sensor_site_id = model.site("sensor").id
    sensor_site_xmat = data.site_xmat[sensor_site_id]

    initial_ee_xpos = data.site_xpos[ee_site_id].copy()
    initial_ee_quat = xmat2quat(data, ee_site_id)

    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]
    data.mocap_pos[mocap_id] = initial_ee_xpos

    mujoco.mj_forward(model, data)

    down_traj = list(np.linspace(initial_ee_xpos[2], 0.06, 2000))
    counter = 0
    hit_target = 0.005
    hit_counter = 0

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    Mx = np.zeros((6, 6))

    force_sensordata = []
    torque_sensordata = []

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=False,
    ) as viewer:
        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY

        while viewer.is_running():
            step_start = time.time()

            if counter != len(down_traj):
                data.mocap_pos[mocap_id][2] = down_traj[counter]
                counter += 1
            else:
                if hit_counter != 1000:
                    data.mocap_pos[mocap_id][2] = hit_target
                    hit_counter += 1
                else:
                    fig, ax = plt.subplots(2, 3)
                    ax[0, 0].plot(
                        np.asarray(range(len(force_sensordata))),
                        np.asarray(force_sensordata)[:, 0],
                    )
                    ax[0, 1].plot(
                        np.asarray(range(len(force_sensordata))),
                        np.asarray(force_sensordata)[:, 1],
                    )
                    ax[0, 2].plot(
                        np.asarray(range(len(force_sensordata))),
                        np.asarray(force_sensordata)[:, 2],
                    )
                    ax[0, 0].set(xlabel="steps", ylabel="x-force")
                    ax[0, 1].set(xlabel="steps", ylabel="y-force")
                    ax[0, 2].set(xlabel="steps", ylabel="z-force")

                    ax[1, 0].plot(
                        np.asarray(range(len(torque_sensordata))),
                        np.asarray(torque_sensordata)[:, 0],
                    )
                    ax[1, 1].plot(
                        np.asarray(range(len(torque_sensordata))),
                        np.asarray(torque_sensordata)[:, 1],
                    )
                    ax[1, 2].plot(
                        np.asarray(range(len(torque_sensordata))),
                        np.asarray(torque_sensordata)[:, 2],
                    )
                    ax[1, 0].set(xlabel="steps", ylabel="x-torque")
                    ax[1, 1].set(xlabel="steps", ylabel="y-torque")
                    ax[1, 2].set(xlabel="steps", ylabel="z-torque")
                    plt.show()
                    break

            dx = data.mocap_pos[mocap_id] - data.site_xpos[ee_site_id].copy()
            twist[:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(site_quat, data.site(ee_site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            # mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_mulQuat(error_quat, initial_ee_quat, site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # Jacobian.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], ee_site_id)

            mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
            Mx_inv = jac @ M_inv @ jac.T
            if abs(np.linalg.det(Mx_inv)) >= 1e-2:
                Mx = np.linalg.inv(Mx_inv)
            else:
                Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

            # Compute generalized forces.
            tau = jac.T @ Mx @ (Kp * twist - Kd * (jac @ data.qvel[dof_ids]))

            # Add joint task in nullspace.
            Jbar = M_inv @ jac.T @ Mx
            # ddq = np.zeros(model.nv)
            # ddq[:6] = Kp_null * (q0 - data.qpos[:6]) - Kd_null * data.qvel[:6]
            ddq = Kp_null * (q0 - data.qpos[dof_ids]) - Kd_null * data.qvel[dof_ids]
            # tau_2 = tau[:6].copy()
            tau += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq

            # Add gravity compensation.
            if gravity_compensation:
                tau += data.qfrc_bias[dof_ids]

            # kp = model.actuator_gainprm[:6, 0]
            # kv = model.actuator_biasprm[:6, 0]
            # Set the control signal and step the simulation.
            # dq = (tau[:6] - kv * data.qvel[:6]) / kp
            # q = data.qpos[:6].copy()
            # ctrl = q + dq

            # np.clip(tau_2, *model.actuator_ctrlrange.T, out=tau_2)
            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            # data.ctrl[:6] = ctrl[:6]
            data.ctrl[actuator_ids] = tau[actuator_ids]
            mujoco.mj_step(model, data, nstep=5)

            force_sensordata.append(data.sensor("ee_force_sensor").data.copy())
            torque_sensordata.append(data.sensor("ee_torque_sensor").data.copy())

            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
