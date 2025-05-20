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
            
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_restitution = True
        restitutions_range = [0.0, 0.4]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] #[-1.0, 1.0] [0.5,0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]# [-1.0, 1.0] [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # [-1, 1] [0.0, 0.0]   # min max [rad/s]
            heading = [-3.14, 3.14] # [-3.14, 3.14] [0.0, 0.0]

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'
        max_iterations = 5000
### ======= TWM =======
class GO2RoughCfgTWM( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_privileged_obs = 57 
        num_envs = 4096
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
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] #[-1.0, 1.0] [0.5,0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]# [-1.0, 1.0] [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # [-1, 1] [0.0, 0.0]   # min max [rad/s]
            heading = [-3.14, 3.14] # [-3.14, 3.14] [0.0, 0.0]
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
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_restitution = True
        restitutions_range = [0.0, 0.4]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

class GO2RoughCfgPPOTWM( LeggedRobotCfgPPO):
    """ Configuration for the GO2 environment with TWM algorithm """
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        
        # transformer algorithm parameters

    class policy( LeggedRobotCfgPPO.policy ):
        imagination_horizon = 8
        
    class twm():
        twm_max_len = 40
        twm_hidden_dim = 64 #256
        twm_num_layers = 2
        twm_num_heads = 8
        twm_train_steps = 1 # train the transformer per this many steps
        twm_start_train_steps = 10 # start training the transformer after this many steps
        twm_start_train_policy_steps = 10# start training the policy using dynamics after this many steps
        twm_train_policy_steps = 5 # train the policy using dynamics per this many steps
        dreaming_batch_size = 1024 # batch size for dreaming 
        batch_length = 32
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
        max_len = 1000000
        BufferWarmUp = 128
        ReplayBufferOnGPU = True
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2_TWM'
        num_steps_per_env = 40 # per iteration
        max_iterations = 5000
### ======= TWM validation =======
class GO2RoughCfgTWM_val( LeggedRobotCfg ):
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
    class env(LeggedRobotCfg.env):
        num_envs = 1
        num_privileged_obs = 57
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [1.0, 1.0] # friction in joint
        randomize_base_mass = True
        added_mass_range = [0.0, 0.0]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
    
    class terrain(LeggedRobotCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0] #[-1.0, 1.0] [0.5,0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # [-1.0, 1.0] [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # [-1, 1] [0.0, 0.0]   # min max [rad/s]
            heading = [-3.14, 3.14] # [-3.14, 3.14] [0.0, 0.0]
            
    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0 # scales other values

class GO2RoughCfgPPOTWM_val( LeggedRobotCfgPPO):
    """ Configuration for the GO2 environment with TWM algorithm """
    seed = 1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        desired_kl = 0.005
        # transformer algorithm parameters

    class policy( LeggedRobotCfgPPO.policy ):
        imagination_horizon = 8
        learning_rate = 1e-5
        
    class twm():
        twm_max_len = 40
        twm_hidden_dim = 64
        twm_num_layers = 4
        twm_num_heads = 8
        twm_train_steps = 2 # train the transformer per this many steps
        twm_start_train_steps = 0 # start training the transformer after this many steps
        twm_start_train_policy_steps = 0 # start training the policy using dynamics after this many steps
        twm_train_policy_steps = 2 # train the policy using dynamics per this many steps
        dreaming_batch_size = 2048 # batch size for dreaming 
        batch_length = 32
        demonstration_batch_size = 0 # batch size for external data
        train_agent_steps = 2 # train the agent this many steps using dynamics
        train_tokenizer_times = 20
        train_dynamic_times = 20
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
        num_steps_per_env = 24 # per iteration
        experiment_name = 'rough_go2_TWM_val'
        max_iterations = 300
### ======= TWM training =======
class GO2RoughCfgTWM_train( LeggedRobotCfg ):
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
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_privileged_obs = 57
        episode_length_s = 100 # episode length in seconds
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_restitution = True
        restitutions_range = [0.0, 0.4]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
    class commands(LeggedRobotCfg.commands):
        class ranges:
            lin_vel_x = [-1.0, 1.0] #[-1.0, 1.0] [0.5,0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # [-1.0, 1.0] [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # [-1, 1] [0.0, 0.0]   # min max [rad/s]
            heading = [-3.14, 3.14] # [-3.14, 3.14] [0.0, 0.0]
    
    class terrain(LeggedRobotCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0 # scales other values

class GO2RoughCfgPPOTWM_train( LeggedRobotCfgPPO):
    """ Configuration for the GO2 environment with TWM algorithm """
    seed = -1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        desired_kl = 0.005
        # transformer algorithm parameters

    class policy( LeggedRobotCfgPPO.policy ):
        imagination_horizon = 8
        learning_rate = 1e-5
        
    class twm():
        twm_max_len = 40
        twm_hidden_dim = 64
        twm_num_layers = 4
        twm_num_heads = 1
        twm_train_steps = 1 # train the transformer per this many steps
        twm_start_train_steps = 0 # start training the transformer after this many steps
        twm_start_train_policy_steps = 0 # start training the policy using dynamics after this many steps
        twm_train_policy_steps = 2 # train the policy using dynamics per this many steps
        dreaming_batch_size = 2048 # batch size for dreaming 
        batch_length = 32
        demonstration_batch_size = 0 # batch size for external data
        train_agent_steps = 2 # train the agent this many steps using dynamics
        train_tokenizer_times = 1
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
        max_len = 100000
        BufferWarmUp = 4096
        ReplayBufferOnGPU = True
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        num_steps_per_env = 100 # per iteration
        experiment_name = 'rough_go2_TWM_train'
        max_iterations = 5000

### ======= GRU TWM training =======
class GO2RoughCfgGRU_train( LeggedRobotCfg ):
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
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_privileged_obs = 57
        episode_length_s = 100 # episode length in seconds
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_restitution = True
        restitutions_range = [0.0, 0.4]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
    class commands(LeggedRobotCfg.commands):
        class ranges:
            lin_vel_x = [-1.0, 1.0] #[-1.0, 1.0] [0.5,0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # [-1.0, 1.0] [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # [-1, 1] [0.0, 0.0]   # min max [rad/s]
            heading = [-3.14, 3.14] # [-3.14, 3.14] [0.0, 0.0]
    
    class terrain(LeggedRobotCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0 # scales other values

class GO2RoughCfgPPOGRU_train( LeggedRobotCfgPPO):
    """ Configuration for the GO2 environment with TWM algorithm """
    seed = 1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        desired_kl = 0.005
        # transformer algorithm parameters

    class policy( LeggedRobotCfgPPO.policy ):
        imagination_horizon = 8
        learning_rate = 1e-5
        
    class gru():
        gru_hidden_size = 256
        mlp_hidden_size = 128
        gru_train_steps = 1 # train the gru per this many steps
        gru_start_train_steps = 0 # start training the gru after this many steps
        gru_start_train_policy_steps = 10 # start training the policy using dynamics after this many steps
        dreaming_batch_size = 4096 # batch size for dreaming 
        batch_length = 32
        demonstration_batch_size = 0 # batch size for external data
        train_agent_steps = 2 # train the agent this many steps using dynamics
        train_tokenizer_times = 0
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
        max_len = 100000
        BufferWarmUp = 4096
        ReplayBufferOnGPU = True
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        num_steps_per_env = 100 # per iteration
        experiment_name = 'rough_go2_GRU_train'
        max_iterations = 5000

### ======= GRU TWM validation =======
class GO2RoughCfgGRU_val( LeggedRobotCfg ):
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
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_privileged_obs = 57
        episode_length_s = 100 # episode length in seconds
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5,0.5]
        randomize_restitution = True
        restitutions_range = [0.1, 0.1]
        randomize_base_mass = True
        added_mass_range = [0.5, 0.5]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.5
    
    class terrain(LeggedRobotCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class commands(LeggedRobotCfg.commands):
        class ranges:
            lin_vel_x = [-1.0, 1.0] #[-1.0, 1.0] [0.5,0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0] # [-1.0, 1.0] [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]   # [-1, 1] [0.0, 0.0]   # min max [rad/s]
            heading = [-3.14, 3.14] # [-3.14, 3.14] [0.0, 0.0]

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0 # scales other values

class GO2RoughCfgPPOGRU_train( LeggedRobotCfgPPO):
    """ Configuration for the GO2 environment with TWM algorithm """
    seed = 1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        desired_kl = 0.005
        # transformer algorithm parameters

    class policy( LeggedRobotCfgPPO.policy ):
        imagination_horizon = 8
        learning_rate = 1e-5
        
    class gru():
        gru_hidden_size = 256
        mlp_hidden_size = 128
        gru_train_steps = 1 # train the gru per this many steps
        gru_start_train_steps = 0 # start training the gru after this many steps
        gru_start_train_policy_steps = 10 # start training the policy using dynamics after this many steps
        dreaming_batch_size = 4096 # batch size for dreaming 
        batch_length = 32
        demonstration_batch_size = 0 # batch size for external data
        train_agent_steps = 2 # train the agent this many steps using dynamics
        train_tokenizer_times = 0
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
        max_len = 100000
        BufferWarmUp = 4096
        ReplayBufferOnGPU = True
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        num_steps_per_env = 400 # per iteration
        experiment_name = 'rough_go2_GRU_train'
        max_iterations = 5000