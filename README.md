# An Optimistic Approach to the Q-Network Error in Actor-Critic Methods
 
This is the repository for the paper _An Optimistic Approach to the Q-Network Error in Actor-Critic Methods_. Source code for the algorithm is found under 
[Algorithms](https://anonymous.4open.science/r/Q-Error-Exploration/Algorithms). 
Learning curves as (1001,) NumPy arrays and their respective figures can be found under [Results](https://anonymous.4open.science/r/Q-Error-Exploration/Results) 
and [Figures](https://anonymous.4open.science/r/Q-Error-Exploration/Figures). 

## Requirements
The following requirements are all publicly available and accessible.
* [Python3](https://www.python.org/downloads/)
* [NumPy](https://numpy.org/) 
* [PyTorch](https://pytorch.org/)
* [OpenAI Gym](https://gym.openai.com/)
* [MuJoCo](https://mujoco.org/)

## Installation
* Download or copy the source files (cloning is not available due to the anonymization).

* Install the dependencies using [requirements.txt](https://anonymous.4open.science/r/Q-Error-Exploration/requirements.txt): 
    ```bash
    pip install -r requirements.txt
    ```
## Run

### DDPG and TD3

1. #### Run:

    ```
    usage: main.py [-h] [--policy POLICY] [--env ENV] [--seed SEED] [--gpu GPU]
               [--start_time_steps N] [--buffer_size BUFFER_SIZE]
               [--eval_freq N] [--max_time_steps N]
               [--exp_regularization EXP_REGULARIZATION]
               [--exploration_noise G] [--batch_size N] [--discount G]
               [--tau G] [--policy_noise G] [--noise_clip G] [--policy_freq N]
               [--save_model] [--load_model LOAD_MODEL]
    ```
  
2. #### Optional Arguments:

    ```
   DDPG, TD3 and their QEX Implementation
   
    optional arguments:
      -h, --help            show this help message and exit
      --policy POLICY       Algorithm (default: QEX_TD3)
      --env ENV             OpenAI Gym environment name
      --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym
                            (default: 0)
      --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
      --start_time_steps N  Number of exploration time steps sampling random
                            actions (default: 1000)
      --buffer_size BUFFER_SIZE
                            Size of the experience replay buffer (default:
                            1000000)
      --eval_freq N         Evaluation period in number of time steps (default:
                            1000)
      --max_time_steps N    Maximum number of steps (default: 1000000)
      --exp_regularization EXP_REGULARIZATION
      --exploration_noise G
                            Std of Gaussian exploration noise
      --batch_size N        Batch size (default: 256)
      --discount G          Discount factor for reward (default: 0.99)
      --tau G               Learning rate in soft/hard updates of the target
                            networks (default: 0.005)
      --policy_noise G      Noise added to target policy during critic update
      --noise_clip G        Range to clip target policy noise
      --policy_freq N       Frequency of delayed policy updates
      --save_model          Save model and optimizer parameters
      --load_model LOAD_MODEL Model load file name; if empty, does not load
    ```
   
### SAC

1. #### Run:

    ```
    usage: main.py [-h] [--policy POLICY] [--policy_type POLICY_TYPE] [--env ENV]
                   [--seed SEED] [--gpu GPU] [--start_steps N]
                   [--exp_regularization EXP_REGULARIZATION]
                   [--buffer_size BUFFER_SIZE] [--eval_freq N] [--num_steps N]
                   [--batch_size N] [--hard_update G] [--train_freq N]
                   [--updates_per_step N] [--target_update_interval N] [--alpha G]
                   [--automatic_entropy_tuning G] [--reward_scale N] [--gamma G]
                   [--tau G] [--lr G] [--hidden_size N]
    ```
  
2. #### Optional Arguments:

    ```
    SAC and its QEX Implementation
    
    optional arguments:
      -h, --help            show this help message and exit
      --policy POLICY       Algorithm (default: QEX_SAC)
      --policy_type POLICY_TYPE
                            Policy Type: Gaussian | Deterministic (default:
                            Gaussian)
      --env ENV             OpenAI Gym environment name
      --seed SEED           Seed number for PyTorch, NumPy and OpenAI Gym
                            (default: 0)
      --gpu GPU             GPU ordinal for multi-GPU computers (default: 0)
      --start_steps N       Number of exploration time steps sampling random
                            actions (default: 1000)
      --exp_regularization EXP_REGULARIZATION
      --buffer_size BUFFER_SIZE
                            Size of the experience replay buffer (default:
                            1000000)
      --eval_freq N         evaluation period in number of time steps (default:
                            1000)
      --num_steps N         Maximum number of steps (default: 1000000)
      --batch_size N        Batch size (default: 256)
      --hard_update G       Hard update the target networks (default: True)
      --train_freq N        Frequency of the training (default: 1)
      --updates_per_step N  Model updates per training time step (default: 1)
      --target_update_interval N
                            Number of critic function updates per training time
                            step (default: 1)
      --alpha G             Temperature parameter α determines the relative
                            importance of the entropy term against the reward
                            (default: 0.2)
      --automatic_entropy_tuning G
                            Automatically adjust α (default: False)
      --reward_scale N      Scale of the environment rewards (default: 5)
      --gamma G             Discount factor for reward (default: 0.99)
      --tau G               Learning rate in soft/hard updates of the target
                            networks (default: 0.005)
      --lr G                Learning rate (default: 0.0003)
      --hidden_size N       Hidden unit size in neural networks (default: 256)
    ```
