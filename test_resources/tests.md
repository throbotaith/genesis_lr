# Tests on prallel simulation in Genesis

> All the tests below are conducted on a desktop with RTX 3080 10GB graphics memory.

## Training performance for a basic task

For a go2 walking on the plane task with 4096 envs, the training speed is approximately **1.3x** compared to [Isaac Gym](https://developer.nvidia.com/isaac-gym).
  
  - Training speed in genesis: 
  ![](./genesis_rl_speed.png)

  - Training speed in isaac gym: 
  ![](./isaacgym_speed.png)
  
While the graphics memory usage is roughly **1/2** compared to IsaacGym.

  - Graphics memory usage in genesis: 
  ![](./genesis_memory_usage.png)

  - Graphics memory usage in isaac gym: 
  ![](./isaacgym_memory_usage.png)

With this smaller memory usage, it's possible to **run more parallel environments**, which can further improve the training speed.

## Attempts on training a decent policy for simulation and deployment

- Simulation
  
  For a go2 walking on the plane task, training a policy with 10000 envs for 600 ites(which is 144M steps) takes about 12mins. The play result is as below:
  
  ![](./go2_flat_play.gif)

- Real Robot
  
  Also for a go2 walking on the plane task, training policy+explicit estimator with 10000 envs for 1k ites takes about 23mins. Deployment result is as below:

  ![](./genesis_deploy_test.gif)

## Heightfield tests

**embedded terrain can't specify difficulty, not practical to use.**

### Time Test

Compilation takes **2min 45s**, with the below params:

| Parameter | Value |
| --- | --- |
| task | go2 |
| headless | False |
| num_envs | 100   |
| horizontal_scale | 0.1 |
| vertical_scale | 0.005 |
| terrain_length | 6.0 |
| terrain_width | 6.0 |
| border_size | 5.0 |
| num_rows | 4 |
| num_cols | 4 |

for headless=True with other params the same, it takes **2min 30s**.

Maybe because that Genesis needs to first compile then execute, it speeds less graphics memory but takes longer time to compile.