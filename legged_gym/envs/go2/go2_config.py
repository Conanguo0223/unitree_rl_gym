from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        # file = '/home/conang/quadruped/unitree_mujoco/unitree_robots/go2/go2.xml'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'

class GO2RoughCfgTWM( LeggedRobotCfgPPO):
    """ Configuration for the GO2 environment with TWM algorithm """
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        
        # transformer algorithm parameters

    class policy( LeggedRobotCfgPPO.policy ):
        imagination_horizon = 8
        
    class twm():
        twm_max_len = 32
        twm_hidden_dim = 512
        twm_num_layers = 2
        twm_num_heads = 8
        twm_train_steps = 2 # train the transformer per this many steps
        twm_start_train_steps = 100 # start training the transformer after this many steps
        twm_start_train_policy_steps = 150 # start training the policy using dynamics after this many steps
        twm_train_policy_steps = 3 # train the policy using dynamics per this many steps
        batch_size = 16
        batch_length = 16
        demonstration_batch_size = 0 # batch size for external data
        train_agent_steps = 5 # train the agent this many steps using dynamics
        train_tokenizer_times = 10
        train_dynamic_times = 10
        use_context = False
        
        # class Agent():
        #     num_layers = 2
        #     hidden_dim = 512
        #     gamma = 0.985
        #     Lambda = 0.95
        #     entropyCoef = 3E-4
        #     use_context = False
    class buffer():
        max_len = 10000
        BufferWarmUp = 128
        ReplayBufferOnGPU = True
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2_TWM'

  
