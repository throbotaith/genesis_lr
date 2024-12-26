# Use terrain

need to use SciPy under 1.14.0(1.13.1), because genesis-world uses inter2p function which is not available since 1.14.0

need to install open3d

embedded terrain can't specify difficulty, not practical to use. 

## Time Test

Compilation takes 2min 45s, with the below params:
| Parameter | Value |
| --- | --- |
| headless | False |
| num_envs | 100   |
| horizontal_scale | 0.1 |
| vertical_scale | 0.005 |
| terrain_length | 6.0 |
| terrain_width | 6.0 |
| border_size | 5.0 |
| num_rows | 4 |
| num_cols | 4 |

for headless=True with other params the same, it takes 2min 30s.