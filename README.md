# Nero-Forte: Reinforcement Learning Module for TurtleBot3 Navigation

Nero-Forte is a reinforcement learning module designed to assist the TurtleBot3 robot in navigating to a specified goal. It uses machine learning techniques to help the robot learn and adapt its navigation strategy.

## How to Use

### Setup

1. Create a workspace and navigate to it:

    ```shell
    mkdir ws && cd ws
    ```

2. Clone the Nero-Forte repository:

    ```shell
    git clone https://github.com/AlphaGotReal/nero-forte.git
    ```

3. Build the project using Catkin:

    ```shell
    rm -rf build devel && catkin_make
    ```

4. Source the workspace:

    ```shell
    source devel/setup.bash
    ```

### Launch Simulation

To launch the simulation environment:

```shell
roslaunch simulation sim.launch
```
To Train and test weights
```shell
rosrun base_chalo driver $NAME_OF_FILE.pth #this is to train the model
rosrun base_chalo $NAME_OF_FILE.pth #this is to test the model
