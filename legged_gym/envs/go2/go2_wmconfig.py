from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
### ======= WM-base config =======
class GO2RoughCfgWM_train(GO2RoughCfg):
    class env(LeggedRobotCfg.env):
        num_privileged_obs = 57 
        num_envs = 4096
    
    class terrain(LeggedRobotCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class commands(LeggedRobotCfg.commands):
        class ranges:
            lin_vel_x = [-1.0, 1.0] #[-1.0, 1.0] [0.5,0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]# [-1.0, 1.0] [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # [-1, 1] [0.0, 0.0]   # min max [rad/s]
            heading = [-3.14, 3.14] # [-3.14, 3.14] [0.0, 0.0]

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

class GO2RoughCfgWM_val(GO2RoughCfg):
    class env(LeggedRobotCfg.env):
        num_privileged_obs = 57 
        num_envs = 1
    
    class terrain(LeggedRobotCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class commands(LeggedRobotCfg.commands):
        class ranges:
            lin_vel_x = [1.0, 1.0] #[-1.0, 1.0] [0.5,0.5] # min max [m/s]
            lin_vel_y = [1.0, 1.0]# [-1.0, 1.0] [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [1.0, 1.0]  # [-1, 1] [0.0, 0.0]   # min max [rad/s]
            heading = [3.14, 3.14] # [-3.14, 3.14] [0.0, 0.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [1.0, 1.0]
        randomize_restitution = True
        restitutions_range = [0.3, 0.3]
        randomize_base_mass = True
        added_mass_range = [1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class noise(LeggedRobotCfg.noise):
        add_noise = False
        noise_level = 1.0 # scales other values

class GO2RoughCfgPPOWM(GO2RoughCfgPPO):
    """ Configuration for the GO2 environment with TWM algorithm """
    seed = -1
    class WM_params():
        context_len = 32
        imagination_horizon = 8
        max_length = 40
        lr = 1e-4
    class wm_training():
        wm_start_train_steps = 10 # start training the transformer after this many steps
        wm_train_steps = 1 # train the transformer per this many steps
        train_wm_times = 10 # train this many times per step
        wm_batch_size = 1024 # batch size for training WM 
    
    class img_policy_training():
        wm_start_train_policy_steps = 10# start training the policy using dynamics after this many steps
        wm_train_policy_steps = 5 # train the policy using dynamics per this many steps
        train_img_policy_times = 10 # train the policy using dynamics this many times per step
        dreaming_batch_size = 2048 # batch size for dreaming

    class buffer():
        max_len = 10000
        BufferWarmUp = 128
        ReplayBufferOnGPU = True

    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 40 # per iteration
        max_iterations = 5000

### ======= transformer WM =======
class GO2RoughCfgTWM＿train(GO2RoughCfgPPOWM):
    class WM_params(GO2RoughCfgPPOWM.WM_params):
        wm_type = 'twm'
        distribution = 'stoch' # stoch
        # transformer world model parameters
        twm_hidden_dim = 128 #256
        twm_num_layers = 2
        twm_num_heads = 2
        batch_length = 32
        latent_feature = 16
        use_context = False
        learning_rate = 1e-3
    
    class runner(GO2RoughCfgPPOWM.runner):
        run_name = ''
        num_steps_per_env = 100 # per iteration(for data collection)
        experiment_name = 'rough_go2_TWM_train'

class GO2RoughCfgPPOTWM_val(GO2RoughCfgTWM＿train):
    """ Configuration for the GO2 environment with TWM algorithm """
    seed = -1
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2_TWM_val'
        max_iterations = 300

### ======= GRU training =======
class GO2RoughCfgGRU_train( GO2RoughCfgPPOWM):
    class WM_params(GO2RoughCfgPPOWM.WM_params):
        wm_type = 'gru'
        gru_hidden_size = 256
        mlp_hidden_size = 128
        use_context = False
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        num_steps_per_env = 100 # per iteration
        experiment_name = 'rough_go2_GRU_train'
        max_iterations = 5000

class GO2RoughCfgPPOGRU_val( GO2RoughCfgGRU_train):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2_GRU_val'
        max_iterations = 300