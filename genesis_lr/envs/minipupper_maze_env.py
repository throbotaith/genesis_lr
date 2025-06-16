from pathlib import Path

import gym
from gym import spaces
import numpy as np
import genesis as gs

class MiniPupperMazeEnv(gym.Env):
    """Simple maze navigation environment for Mini Pupper 2 using Genesis."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, headless=True):
        super().__init__()
        gs.init(backend=gs.gpu, logging_level='warning')

        repo_root = Path(__file__).resolve().parents[2]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.02, substeps=1),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=50,
                camera_pos=(2.0, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.3),
                camera_fov=40,
            ),
            show_viewer=not headless,
        )

        # load robot (use go2 as placeholder for mini pupper)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(repo_root / "resources" / "robots" / "go2" / "urdf" / "go2.urdf"),
                merge_fixed_links=True,
                pos=(0.0, 0.0, 0.24),
            ),
        )

        # camera attached to the robot base
        self.camera = self.scene.add_camera(
            res=(64, 64),
            pos=(0.2, 0.0, 0.15),
            lookat=(1.0, 0.0, 0.15),
            fov=90,
            GUI=False,
        )
        self._update_camera()

        # build a simple maze with two walls
        h = 0.2
        w = 0.05
        l = 4.0
        self.scene.add_entity(gs.morphs.Box(size=(l, w, h), pos=(1.0, 0.5, h/2), fixed=True))
        self.scene.add_entity(gs.morphs.Box(size=(l, w, h), pos=(1.0, -0.5, h/2), fixed=True))
        self.goal_pos = np.array([2.0, 0.0])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8)
        self.step_count = 0

    def _update_camera(self):
        pos = self.robot.get_pos()
        quat = self.robot.get_quat()
        offset = gs.transform_by_quat(np.array([0.2, 0.0, 0.15]), quat)
        look = gs.transform_by_quat(np.array([1.0, 0.0, 0.15]), quat)
        self.camera.set_pose(pos=pos + offset, lookat=pos + look)

    def _get_obs(self):
        img = self.camera.get_rgba_tensor()[..., :3].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        return np.transpose(img, (2, 0, 1))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.set_pos((0.0, 0.0, 0.24))
        self.robot.set_quat((1.0, 0.0, 0.0, 0.0))
        self.step_count = 0
        self._update_camera()
        return self._get_obs(), {}

    def step(self, action):
        # simple discrete control using base velocity
        if action == 0:
            vel = (0.3, 0.0, 0.0)
            yaw = 0.0
        elif action == 1:
            vel = (0.0, 0.0, 0.0)
            yaw = 1.0
        else:
            vel = (0.0, 0.0, 0.0)
            yaw = -1.0
        self.robot.set_base_velocity(linear=vel, angular=(0.0, 0.0, yaw))

        self.scene.step()
        self._update_camera()

        obs = self._get_obs()
        base = self.robot.get_pos()
        dist = np.linalg.norm(base[:2] - self.goal_pos)
        reward = -dist
        done = dist < 0.2
        self.step_count += 1
        if self.step_count >= 500:
            done = True
        return obs, reward, done, {}

    def render(self):
        pass
