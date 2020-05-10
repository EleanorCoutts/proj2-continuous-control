# Reacher - README

The code in this repository trains a deep reinforcement learning agent to solve the Unity Environment Reacher, where a double jointed arm needs to keep its' 'hand' in the target location for as long as possible. This completes Project 2 of Udacity's Deep Reinforcement Learning Nanodegree. <br>

A single copy of this environment was used to train the agent. The agent trained used an actor-critic method - the DDPG algorithm as, described in Lillicrap et.al 2015. The agent learns a deterministic policy that gives continuous action values. A descriptioin of the algorithm as parameters used is in the Report in this repository.



## Training the agent

Download a local copy of this repository.

### Setup the environment.

To get the environment running on your local machine, follow the following steps (as in the Udacity Project Intructions).

Download the environment from one of the links below.  You need only select the environment that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Unzip and save the environment in your local repository, and change the path in `Continuous_Control.ipynb` to match your local file structure.

### Run the code

Run all the code in `Continuous_Control.ipynb` to train the agent. Note that the code will save information in the `checkpoints` folder every 20 episodes by default (specified by the `SAVE_EVERY` parameter). Files saved are the network parameters for both the critic and actor, and the score history. Filenames contain the timestamp of when the training process started, and also the episode number at that checkpoint. When training is complete the same information is saved with the word `Final` in the filename. <br> 

Note that in this repository only the final checkpoint is stored.

## Load pre-trained agent

A pre-trained agent is saved in the `checkpoints` folder. The file `20200509_132504_actor_Final.pth` contains the paramaters of the trained actor, the file `20200509_132504_critic_Final.pth` stores parameters of the trained critic. Use `torch.save` to load these pre-trained agents instead of training from scratch.
