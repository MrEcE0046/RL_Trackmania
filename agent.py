import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pydirectinput
import cv2
import mss
import math

class TrackmaniaEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()

        self.num_rays = 5
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_rays + 1,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # 0: rakt, 1: vänster, 2: höger, 3: coast, 4: broms

        self.region = {
            "top": 100,
            "left": 100,
            "width": 800,
            "height": 600
        }
        
        self.origin = (400, 380)
        self.radar_angles = [-180, -150, -90, -30, 0]

        self.prev_position = self.origin
        self.crash_counter = 0
        self.max_crashes = 3

    def reset(self, seed=None, options=None):
        self._reset_to_start()
        self.prev_position = self.origin
        self.crash_counter = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self._send_action(action)
        time.sleep(0.1)

        obs = self._get_observation()
        reward = self._calculate_reward(obs)
        terminated = self._check_done(obs)

        print(f"Reward: {reward:.3f}, Speed: {obs[-1]:.2f}")

        return obs, reward, terminated, False, {}

    def _get_observation(self):
        frame = self._capture_screen(self.region)

        curr_pos = self._find_car_position(frame)

        if curr_pos is None:
            speed = 0.0
            radar = self._get_radar_readings(frame, self.prev_position, self.radar_angles)
        else:
            dx = curr_pos[0] - self.prev_position[0]
            dy = curr_pos[1] - self.prev_position[1]
            speed = np.clip(np.sqrt(dx ** 2 + dy ** 2) / 50.0, 0.0, 1.0)
            self.prev_position = curr_pos
            radar = self._get_radar_readings(frame, curr_pos, self.radar_angles)

        return np.append(radar, speed).astype(np.float32)

    def _find_car_position(self, image):
        """Hittar bilens position med färgdetektering (röd)."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_color = np.array([0, 100, 100])
        upper_color = np.array([10, 255, 255])
        
        mask = cv2.inRange(hsv_image, lower_color, upper_color)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        
        return None

    def _send_action(self, action):
        pydirectinput.keyUp("left")
        pydirectinput.keyUp("right")
        pydirectinput.keyUp("up")
        pydirectinput.keyUp("down")

        if action in [0, 1, 2]:
            pydirectinput.keyDown("up")

        if action == 1:
            pydirectinput.keyDown("left")
        elif action == 2:
            pydirectinput.keyDown("right")
        elif action == 4:
            pydirectinput.keyDown("down")

    def _calculate_reward(self, obs):
        radar = obs[:-1]
        speed = obs[-1]
        wall_penalty = -1.0 * (1.0 - np.min(radar))
        reward = speed + wall_penalty

        if speed < 0.01:
            self.crash_counter += 1
            if self.crash_counter >= self.max_crashes:
                reward -= 3.0
        else:
            self.crash_counter = 0
        
        return float(reward)

    def _check_done(self, obs):
        speed = obs[-1]
        if speed < 0.01:
            self.crash_counter += 1
        else:
            self.crash_counter = 0

        return self.crash_counter >= self.max_crashes

    def _reset_to_start(self):
        pydirectinput.keyUp("up")
        pydirectinput.keyUp("left")
        pydirectinput.keyUp("right")
        pydirectinput.keyUp("down")
        time.sleep(0.2)
        pydirectinput.press("backspace")
        time.sleep(1.0)
    
    # --- Inkluderade funktioner från screen_capture.py och radar_test.py ---
    
    def _capture_screen(self, region=None):
        with mss.mss() as sct:
            monitor = region if region else sct.monitors[1]
            img = np.array(sct.grab(monitor))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _is_wall(self, pixel):
        return np.mean(pixel) < 50

    def _cast_ray(self, image, origin, angle_deg, max_distance=150):
        angle_rad = math.radians(angle_deg)
        x0, y0 = origin
        for d in range(max_distance):
            x = int(x0 + d * math.cos(angle_rad))
            y = int(y0 + d * math.sin(angle_rad))
            if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                return d / max_distance
            pixel = image[y, x]
            if self._is_wall(pixel):
                return d / max_distance
        return 1.0

    def _get_radar_readings(self, image, origin, angles):
        readings = []
        for angle in angles:
            dist = self._cast_ray(image, origin, angle)
            readings.append(dist)
        return np.array(readings)

# Om du vill köra en enskild test av miljön

    # env = TrackmaniaEnv()
    # obs, info = env.reset()
    # for _ in range(100):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated:
    #         print("Episode terminated!")
    #         break
    # env.close()