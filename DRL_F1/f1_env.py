# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# import pygame


# class F1Env(gym.Env):
#     """
#     F1-style top-down racing environment with a CarRacing-like interface.

#     - Observation: (96, 96, 3) uint8 RGB, ego-centric top-down
#     - Action: [steer, gas, brake]
#         steer in [-1, 1]
#         gas   in [0, 1]
#         brake in [0, 1]
#     """

#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

#     def __init__(self, render_mode=None):
#         super().__init__()

#         # ----- Spaces -----
#         self.observation_space = spaces.Box(
#             low=0,
#             high=255,
#             shape=(96, 96, 3),
#             dtype=np.uint8,
#         )

#         self.action_space = spaces.Box(
#             low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
#             high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
#             dtype=np.float32,
#         )

#         # ----- Dynamics -----
#         self.dt = 0.05           # 20 Hz
#         self.max_speed = 65.0    # m/s, F1-ish top speed
#         self.min_speed = 0.0
#         self.max_steer = np.deg2rad(28.0)
#         self.max_accel = 18.0
#         self.max_brake = 35.0
#         self.drag = 1.4
#         self.L = 3.6             # wheelbase

#         # Track parameters
#         self.track_width = 10.0
#         self.num_waypoints = 300
#         self.waypoints = self._create_f1_track(self.num_waypoints)

#         # State: [x, y, heading, speed, progress_idx]
#         self.state = None

#         # Rendering
#         self.render_mode = render_mode
#         self.screen = None
#         self.clock = None
#         self.window_size = 512  # pygame window size

#     # ===================== TRACK GEOMETRY ===================== #

#     def _create_f1_track(self, n_points):
#         """
#         Construct a simple F1-style circuit using a polyline of control points,
#         then resample it to `n_points` waypoints.
#         The shape: long main straight, hairpin, flowing S-bends, chicane.
#         """
#         # Control points in world coordinates (meters)
#         cps = np.array([
#             # main straight (bottom, left -> right)
#             [-60.0, -40.0],
#             [-20.0, -40.0],
#             [ 20.0, -40.0],
#             [ 60.0, -40.0],
#             [ 80.0, -20.0],   # braking zone up
#             [ 70.0,   0.0],
#             [ 40.0,  25.0],   # sweeping right
#             [ 10.0,  35.0],
#             [-20.0,  35.0],
#             [-50.0,  25.0],
#             [-70.0,   5.0],
#             [-60.0, -20.0],
#             [-60.0, -40.0],   # back to start
#         ], dtype=np.float32)

#         # Compute cumulative arc length along control polyline
#         diffs = cps[1:] - cps[:-1]
#         seg_lens = np.linalg.norm(diffs, axis=1)
#         cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
#         total_len = cum_lens[-1]

#         # Target positions along the track
#         s_vals = np.linspace(0.0, total_len, n_points, endpoint=False)

#         # Resample points
#         waypoints = []
#         for s in s_vals:
#             # Find which segment s is in
#             idx = np.searchsorted(cum_lens, s, side="right") - 1
#             idx = np.clip(idx, 0, len(seg_lens) - 1)
#             s0 = cum_lens[idx]
#             t = (s - s0) / max(seg_lens[idx], 1e-6)
#             p = cps[idx] * (1.0 - t) + cps[idx + 1] * t
#             waypoints.append(p)

#         return np.array(waypoints, dtype=np.float32)

#     def _closest_waypoint_idx(self, position):
#         diffs = self.waypoints - position[None, :]
#         dists = np.linalg.norm(diffs, axis=-1)
#         return int(np.argmin(dists))

#     # ========================= GYM API ======================== #

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)

#         # Start near beginning of main straight, heading along +x
#         start_idx = 0
#         start_pos = self.waypoints[start_idx]
#         heading = 0.0
#         speed = 20.0  # rolling start
#         progress_idx = float(start_idx)

#         self.state = np.array(
#             [start_pos[0], start_pos[1], heading, speed, progress_idx],
#             dtype=np.float32,
#         )

#         obs = self._get_observation()
#         info = {"progress_idx": progress_idx}
#         return obs, info

#     def step(self, action):
#         action = np.clip(action, self.action_space.low, self.action_space.high)

#         steer, gas, brake = float(action[0]), float(action[1]), float(action[2])

#         x, y, heading, speed, progress_idx = self.state

#         # ---- Dynamics ----
#         # steering angle
#         delta = self.max_steer * steer

#         # longitudinal acceleration
#         accel = gas * self.max_accel - brake * self.max_brake - self.drag * speed
#         speed = np.clip(speed + accel * self.dt, self.min_speed, self.max_speed)

#         # kinematics (bicycle model)
#         beta = np.arctan(0.5 * np.tan(delta))
#         x = x + speed * np.cos(heading + beta) * self.dt
#         y = y + speed * np.sin(heading + beta) * self.dt
#         heading = heading + (speed / max(self.L, 1e-3)) * np.tan(delta) * self.dt

#         position = np.array([x, y], dtype=np.float32)
#         wp_idx = self._closest_waypoint_idx(position)

#         # progress around track
#         prev_idx = int(progress_idx)
#         delta_idx = (wp_idx - prev_idx) % self.num_waypoints
#         progress_idx = (progress_idx + delta_idx) % self.num_waypoints

#         # ---- Reward ----
#         # reward for moving forward along track
#         progress_reward = float(delta_idx)

#         # distance from centerline
#         center = self.waypoints[wp_idx]
#         dist_from_center = float(np.linalg.norm(position - center))

#         offtrack = dist_from_center > (self.track_width / 2.0)
#         offtrack_penalty = -6.0 if offtrack else 0.0

#         # mild speed shaping: encourage ~40–55 m/s, penalize too slow/too fast
#         speed_target = 50.0
#         speed_penalty = -0.01 * abs(speed - speed_target)

#         time_penalty = -0.01

#         reward = progress_reward + offtrack_penalty + speed_penalty + time_penalty

#         # ---- Termination / truncation ----
#         terminated = False
#         if dist_from_center > (self.track_width * 2.5):
#             terminated = True
#             reward -= 25.0  # big crash penalty

#         truncated = False  # you can add a max steps wrapper externally

#         self.state = np.array([x, y, heading, speed, progress_idx], dtype=np.float32)

#         obs = self._get_observation()
#         info = {
#             "progress_idx": progress_idx,
#             "dist_from_center": dist_from_center,
#             "speed": speed,
#         }

#         return obs, reward, terminated, truncated, info

#     # ===================== RENDERING ========================== #

#     def _get_observation(self):
#         """
#         Ego-centric camera like CarRacing:
#         - Car is near bottom center of image (fixed).
#         - World (track) scrolls around it.
#         - Grass background, dark asphalt road, white dashed centerline.
#         """
#         H, W = 96, 96
#         surface = pygame.Surface((W, H))

#         # Grass background
#         grass_color = (34, 139, 34)  # a bit darker green
#         surface.fill(grass_color)

#         x, y, heading, speed, progress_idx = self.state

#         # Camera parameters: car fixed at (cx, cy)
#         cx = W / 2.0
#         cy = H * 0.75          # car near bottom
#         scale = 1.0 / 1.2      # pixels per meter, tune if needed

#         # world -> ego rotation (so car faces "up")
#         cos_h = np.cos(-heading)
#         sin_h = np.sin(-heading)

#         def world_to_ego(px, py):
#             dx = px - x
#             dy = py - y
#             ex = cos_h * dx - sin_h * dy
#             ey = sin_h * dx + cos_h * dy
#             return ex, ey

#         def ego_to_img(ex, ey):
#             ix = int(cx + ex * scale)
#             iy = int(cy - ey * scale)
#             return ix, iy

#         # --- draw track as dark asphalt strip with dashed white center line ---
#         asphalt_color = (45, 45, 50)
#         centerline_color = (230, 230, 230)

#         track_w_pixels = max(int(self.track_width * scale), 4)  # road width in pixels
#         centerline_w = 2

#         for i in range(self.num_waypoints):
#             p1 = self.waypoints[i]
#             p2 = self.waypoints[(i + 1) % self.num_waypoints]

#             ex1, ey1 = world_to_ego(p1[0], p1[1])
#             ex2, ey2 = world_to_ego(p2[0], p2[1])

#             # Skip segments that are far away (for speed)
#             mx = 0.5 * (ex1 + ex2)
#             my = 0.5 * (ey1 + ey2)
#             if abs(mx) > 80 or abs(my) > 80:
#                 continue

#             ix1, iy1 = ego_to_img(ex1, ey1)
#             ix2, iy2 = ego_to_img(ex2, ey2)

#             # Asphalt strip
#             pygame.draw.line(
#                 surface,
#                 asphalt_color,
#                 (ix1, iy1),
#                 (ix2, iy2),
#                 track_w_pixels,
#             )

#             # Dashed white center line: draw every few segments
#             if i % 4 == 0:
#                 pygame.draw.line(
#                     surface,
#                     centerline_color,
#                     (ix1, iy1),
#                     (ix2, iy2),
#                     centerline_w,
#                 )

#         # Convert surface → numpy (H, W, 3)
#         img = pygame.surfarray.array3d(surface).swapaxes(0, 1)

#         # Draw the car at fixed ego position (cx, cy)
#         self._draw_f1_car_sprite(img, int(cx), int(cy), heading=0.0)

#         return img

#     def _draw_f1_car_sprite(self, img, cx, cy, heading=0.0):
#         """
#         Small F1-ish sprite: body + front & rear wings + cockpit.
#         Sized to look similar to CarRacing's tiny car.
#         """
#         H, W, _ = img.shape

#         body_color = np.array([10, 10, 10], dtype=np.uint8)       # black
#         wing_color = np.array([240, 240, 240], dtype=np.uint8)    # white
#         cockpit_color = np.array([40, 60, 200], dtype=np.uint8)   # blue

#         # Body (thin vertical rectangle)
#         body_w = 3
#         body_h = 6
#         x_min = max(cx - body_w // 2, 0)
#         x_max = min(cx + body_w // 2, W - 1)
#         y_min = max(cy - body_h, 0)
#         y_max = min(cy, H - 1)
#         img[y_min:y_max+1, x_min:x_max+1] = body_color

#         # Front wing (small horizontal bar)
#         fw_w = 6
#         fw_h = 1
#         fw_y = max(y_min - fw_h, 0)
#         fw_x_min = max(cx - fw_w // 2, 0)
#         fw_x_max = min(cx + fw_w // 2, W - 1)
#         img[fw_y:y_min, fw_x_min:fw_x_max+1] = wing_color

#         # Rear wing (shorter bar behind car)
#         rw_w = 5
#         rw_h = 1
#         rw_y0 = min(y_max + 1, H - 1)
#         rw_y1 = min(rw_y0 + rw_h, H - 1)
#         rw_x_min = max(cx - rw_w // 2, 0)
#         rw_x_max = min(cx + rw_w // 2, W - 1)
#         img[rw_y0:rw_y1+1, rw_x_min:rw_x_max+1] = wing_color

#         # Cockpit (tiny blue dot near front)
#         cp_y = max(y_min + 1, 0)
#         if 0 <= cx < W and 0 <= cp_y < H:
#             img[cp_y, cx] = cockpit_color

    
#     # ===================== EXTERNAL RENDER ==================== #

#     def render(self):
#         if self.render_mode is None:
#             return

#         img = self._get_observation()

#         if self.render_mode == "rgb_array":
#             return img

#         if self.render_mode == "human":
#             if self.screen is None:
#                 pygame.init()
#                 self.screen = pygame.display.set_mode((self.window_size, self.window_size))
#                 self.clock = pygame.time.Clock()

#             surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
#             surf = pygame.transform.scale(surf, (self.window_size, self.window_size))
#             self.screen.blit(surf, (0, 0))
#             pygame.display.flip()
#             self.clock.tick(self.metadata["render_fps"])

#     def close(self):
#         if self.screen is not None:
#             pygame.quit()
#             self.screen = None
#             self.clock = None


import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame


class F1Env(gym.Env):
    """
    Custom F1-style racing environment with a CarRacing-like interface.

    Observation:
        (96, 96, 3) uint8 RGB, CarRacing-style ego view
    Action:
        [steer, gas, brake]
            steer in [-1, 1]
            gas   in [0, 1]
            brake in [0, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        # ----- Spaces -----
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(96, 96, 3),
            dtype=np.uint8,
        )

        # steer, gas, brake
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ----- Dynamics -----
        self.dt = 0.05           # 20 Hz
        self.max_speed = 65.0    # m/s
        self.min_speed = 0.0
        self.max_steer = np.deg2rad(28.0)
        self.max_accel = 18.0
        self.max_brake = 35.0
        self.drag = 1.4
        self.L = 3.6             # wheelbase

        # Track parameters
        self.track_width = 10.0
        self.num_waypoints = 300
        self.waypoints = self._create_f1_track(self.num_waypoints)

        # State: [x, y, heading, speed, progress_idx]
        self.state = None

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.window_size = 512  # pygame window size

    # ===================== TRACK GEOMETRY ===================== #

    def _create_f1_track(self, n_points: int) -> np.ndarray:
        """
        Construct a simple F1-style circuit using a polyline of control points,
        then resample it to `n_points` waypoints.

        Shape: long main straight, fast sweeper, tighter section, hairpin.
        """
        cps = np.array([
            # main straight (bottom, left -> right)
            [-70.0, -40.0],
            [-20.0, -40.0],
            [ 20.0, -40.0],
            [ 70.0, -40.0],
            [ 90.0, -10.0],   # braking up
            [ 70.0,  20.0],
            [ 40.0,  40.0],   # long right
            [  5.0,  50.0],
            [-25.0,  45.0],
            [-55.0,  25.0],
            [-75.0,   0.0],
            [-70.0, -25.0],
            [-70.0, -40.0],   # close loop
        ], dtype=np.float32)

        diffs = cps[1:] - cps[:-1]
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cum_lens[-1]

        s_vals = np.linspace(0.0, total_len, n_points, endpoint=False)

        waypoints = []
        for s in s_vals:
            idx = np.searchsorted(cum_lens, s, side="right") - 1
            idx = np.clip(idx, 0, len(seg_lens) - 1)
            s0 = cum_lens[idx]
            t = (s - s0) / max(seg_lens[idx], 1e-6)
            p = cps[idx] * (1.0 - t) + cps[idx + 1] * t
            waypoints.append(p)

        return np.array(waypoints, dtype=np.float32)

    def _closest_waypoint_idx(self, position: np.ndarray) -> int:
        diffs = self.waypoints - position[None, :]
        dists = np.linalg.norm(diffs, axis=-1)
        return int(np.argmin(dists))

    # ========================= GYM API ======================== #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start_idx = 0
        start_pos = self.waypoints[start_idx]

        # Heading along first segment direction
        next_pos = self.waypoints[(start_idx + 1) % self.num_waypoints]
        heading = float(np.arctan2(next_pos[1] - start_pos[1],
                                   next_pos[0] - start_pos[0]))

        speed = 20.0  # rolling start
        progress_idx = float(start_idx)

        self.state = np.array(
            [start_pos[0], start_pos[1], heading, speed, progress_idx],
            dtype=np.float32,
        )

        obs = self._get_observation()
        info = {"progress_idx": progress_idx}
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        steer, gas, brake = float(action[0]), float(action[1]), float(action[2])
        x, y, heading, speed, progress_idx = self.state

        # ---- Dynamics ----
        delta = self.max_steer * steer
        accel = gas * self.max_accel - brake * self.max_brake - self.drag * speed
        speed = np.clip(speed + accel * self.dt, self.min_speed, self.max_speed)

        # bicycle model
        x = x + speed * np.cos(heading) * self.dt
        y = y + speed * np.sin(heading) * self.dt
        heading = heading + (speed / max(self.L, 1e-3)) * np.tan(delta) * self.dt

        position = np.array([x, y], dtype=np.float32)
        wp_idx = self._closest_waypoint_idx(position)

        prev_idx = int(progress_idx)
        delta_idx = (wp_idx - prev_idx) % self.num_waypoints
        progress_idx = (progress_idx + delta_idx) % self.num_waypoints

        # ---- Reward ----
        progress_reward = float(delta_idx)

        center = self.waypoints[wp_idx]
        dist_from_center = float(np.linalg.norm(position - center))
        offtrack = dist_from_center > (self.track_width / 2.0)
        offtrack_penalty = -6.0 if offtrack else 0.0

        # speed shaping around target
        speed_target = 50.0
        speed_penalty = -0.01 * abs(speed - speed_target)

        time_penalty = -0.01

        reward = progress_reward + offtrack_penalty + speed_penalty + time_penalty

        terminated = False
        if dist_from_center > (self.track_width * 2.5):
            terminated = True
            reward -= 25.0  # big crash penalty

        truncated = False  # can wrap with TimeLimit externally

        self.state = np.array([x, y, heading, speed, progress_idx], dtype=np.float32)

        obs = self._get_observation()
        info = {
            "progress_idx": progress_idx,
            "dist_from_center": dist_from_center,
            "speed": speed,
        }

        return obs, reward, terminated, truncated, info

    # ===================== RENDERING ========================== #

    def _get_observation(self):
        """
        Ego-centric camera like CarRacing:
        - Car fixed near bottom centre.
        - Dark asphalt road with soft edges and dashed centre line.
        """
        H, W = 96, 96
        surface = pygame.Surface((W, H))

        # Grass background with light noise
        base_grass = np.array([34, 139, 34], dtype=np.uint8)
        grass = np.clip(
            base_grass[None, None, :] +
            np.random.randint(-5, 6, size=(H, W, 3)),
            0, 255,
        ).astype(np.uint8)
        pygame.surfarray.blit_array(surface, grass.swapaxes(0, 1))

        x, y, heading, speed, progress_idx = self.state

        # Camera: car fixed at (cx, cy)
        cx = W / 2.0
        cy = int(H * 0.75)
        scale = 1.0 / 1.3   # pixels per meter

        cos_h = np.cos(-heading)
        sin_h = np.sin(-heading)

        def world_to_ego(px, py):
            dx = px - x
            dy = py - y
            ex = cos_h * dx - sin_h * dy
            ey = sin_h * dx + cos_h * dy
            return ex, ey

        def ego_to_img(ex, ey):
            ix = int(cx + ex * scale)
            iy = int(cy - ey * scale)
            return ix, iy

        asphalt_color    = (45, 45, 50)
        shoulder_color   = (70, 70, 75)
        centerline_color = (230, 230, 230)

        track_w = max(int(self.track_width * scale), 5)
        shoulder_w = max(track_w - 2, 3)
        center_w = 2

        for i in range(self.num_waypoints):
            p1 = self.waypoints[i]
            p2 = self.waypoints[(i + 1) % self.num_waypoints]

            ex1, ey1 = world_to_ego(p1[0], p1[1])
            ex2, ey2 = world_to_ego(p2[0], p2[1])

            # skip far segments
            mx = 0.5 * (ex1 + ex2)
            my = 0.5 * (ey1 + ey2)
            if abs(mx) > 80 or abs(my) > 80:
                continue

            ix1, iy1 = ego_to_img(ex1, ey1)
            ix2, iy2 = ego_to_img(ex2, ey2)

            # Outer dark asphalt
            pygame.draw.line(
                surface, asphalt_color,
                (ix1, iy1), (ix2, iy2),
                track_w,
            )
            # Slightly lighter inner band
            pygame.draw.line(
                surface, shoulder_color,
                (ix1, iy1), (ix2, iy2),
                shoulder_w,
            )

            # Dashed white centre line
            if i % 3 == 0:
                pygame.draw.line(
                    surface, centerline_color,
                    (ix1, iy1), (ix2, iy2),
                    center_w,
                )

        img = pygame.surfarray.array3d(surface).swapaxes(0, 1)

        # Draw car at fixed ego position
        self._draw_f1_car_sprite(img, cx, cy, heading=0.0)

        return img

    def _draw_f1_car_sprite(self, img, cx, cy, heading=0.0):
        """
        Small F1-ish sprite: body + wings + cockpit.
        Sized similar to CarRacing's car.
        """
        H, W, _ = img.shape

        body_color    = np.array([10, 10, 10], dtype=np.uint8)       # black
        wing_color    = np.array([240, 240, 240], dtype=np.uint8)    # white
        cockpit_color = np.array([40, 60, 200], dtype=np.uint8)      # blue

        # Clamp centre
        cx = int(np.clip(cx, 0, W - 1))
        cy = int(np.clip(cy, 0, H - 1))

        # Body (vertical)
        body_w = 3
        body_h = 6
        x_min = max(cx - body_w // 2, 0)
        x_max = min(cx + body_w // 2, W - 1)
        y_min = max(cy - body_h, 0)
        y_max = min(cy, H - 1)
        img[y_min:y_max+1, x_min:x_max+1] = body_color

        # Front wing
        fw_w = 6
        fw_h = 1
        fw_y0 = max(y_min - fw_h, 0)
        fw_y1 = y_min
        fw_x_min = max(cx - fw_w // 2, 0)
        fw_x_max = min(cx + fw_w // 2, W - 1)
        img[fw_y0:fw_y1, fw_x_min:fw_x_max+1] = wing_color

        # Rear wing
        rw_w = 5
        rw_h = 1
        rw_y0 = min(y_max + 1, H - 1)
        rw_y1 = min(rw_y0 + rw_h, H - 1)
        rw_x_min = max(cx - rw_w // 2, 0)
        rw_x_max = min(cx + rw_w // 2, W - 1)
        img[rw_y0:rw_y1+1, rw_x_min:rw_x_max+1] = wing_color

        # Cockpit (blue dot)
        cp_y = max(y_min + 1, 0)
        if 0 <= cx < W and 0 <= cp_y < H:
            img[cp_y, cx] = cockpit_color

    # ===================== EXTERNAL RENDER ==================== #

    def render(self):
        if self.render_mode is None:
            return

        img = self._get_observation()

        if self.render_mode == "rgb_array":
            return img

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.window_size, self.window_size))
                self.clock = pygame.time.Clock()

            surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            surf = pygame.transform.scale(surf, (self.window_size, self.window_size))
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
