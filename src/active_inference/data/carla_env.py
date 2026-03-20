"""CARLA driving environment wrapper."""

from __future__ import annotations

import queue

import numpy as np

try:
    import carla

    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


def action_to_carla(action_2d: np.ndarray) -> tuple[float, float, float]:
    steer = float(np.clip(action_2d[0], -1.0, 1.0))
    accel = float(action_2d[1])
    if accel >= 0:
        throttle = float(np.clip(accel, 0.0, 1.0))
        brake = 0.0
    else:
        throttle = 0.0
        brake = float(np.clip(-accel, 0.0, 1.0))
    return steer, throttle, brake


def carla_to_action(steer: float, throttle: float, brake: float) -> np.ndarray:
    s = float(np.clip(steer, -1.0, 1.0))
    accel = float(np.clip(throttle - brake, -1.0, 1.0))
    return np.array([s, accel], dtype=np.float32)


class CARLADrivingEnv:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        town: str = "Town06",
        image_capture_size: int = 256,
        image_model_size: int = 64,
        fov: int = 90,
    ):
        if not CARLA_AVAILABLE:
            raise RuntimeError("carla package is not installed.")

        self.host = host
        self.port = port
        self.town = town
        self.image_capture_size = image_capture_size
        self.image_model_size = image_model_size
        self.fov = fov

        self._client = carla.Client(host, port)
        self._client.set_timeout(60.0)
        self._world = self._client.load_world(town)
        self._client.set_timeout(10.0)

        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self._world.apply_settings(settings)

        self._vehicle = None
        self._camera = None
        self._chase_camera = None
        self._collision_sensor = None
        self._lane_sensor = None
        self._image_queue: queue.Queue = queue.Queue(maxsize=1)
        self._chase_image_queue: queue.Queue = queue.Queue(maxsize=1)
        self._collision_flag = False
        self._lane_invasion_flag = False
        self._goal_location = None

    def reset(self, spawn_point=None):
        self._cleanup_actors()
        self._collision_flag = False
        self._lane_invasion_flag = False

        blueprint_library = self._world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.lincoln.mkz_2020")

        if spawn_point is None:
            spawn_points = self._world.get_map().get_spawn_points()
            spawn_point = np.random.choice(spawn_points)

        self._vehicle = self._world.spawn_actor(vehicle_bp, spawn_point)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_capture_size))
        camera_bp.set_attribute("image_size_y", str(self.image_capture_size))
        camera_bp.set_attribute("fov", str(self.fov))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self._camera = self._world.spawn_actor(camera_bp, camera_transform, attach_to=self._vehicle)
        self._camera.listen(self._on_image)

        collision_bp = blueprint_library.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self._vehicle
        )
        self._collision_sensor.listen(self._on_collision)

        lane_bp = blueprint_library.find("sensor.other.lane_invasion")
        self._lane_sensor = self._world.spawn_actor(
            lane_bp, carla.Transform(), attach_to=self._vehicle
        )
        self._lane_sensor.listen(self._on_lane_invasion)

        self._setup_chase_camera()

        self._world.tick()
        return self._get_observation()

    def _setup_chase_camera(self):
        if not CARLA_AVAILABLE or self._vehicle is None:
            return
        blueprint_library = self._world.get_blueprint_library()
        chase_bp = blueprint_library.find("sensor.camera.rgb")
        chase_bp.set_attribute("image_size_x", "960")
        chase_bp.set_attribute("image_size_y", "540")
        chase_bp.set_attribute("fov", "90")
        chase_transform = carla.Transform(
            carla.Location(x=-8.0, z=5.0), carla.Rotation(pitch=-15.0)
        )
        self._chase_camera = self._world.spawn_actor(
            chase_bp, chase_transform, attach_to=self._vehicle
        )
        self._chase_camera.listen(self._on_chase_image)

    def _on_chase_image(self, image):
        if not self._chase_image_queue.full():
            self._chase_image_queue.put_nowait(image)
        else:
            try:
                self._chase_image_queue.get_nowait()
            except queue.Empty:
                pass
            self._chase_image_queue.put_nowait(image)

    def get_chase_frame(self):
        try:
            raw = self._chase_image_queue.get(timeout=2.0)
        except queue.Empty:
            return None
        array = np.frombuffer(raw.raw_data, dtype=np.uint8)
        array = array.reshape((raw.height, raw.width, 4))
        return array[:, :, :3][:, :, ::-1].copy()

    def step(self, action_2d: np.ndarray):
        self._collision_flag = False
        self._lane_invasion_flag = False

        steer, throttle, brake = action_to_carla(action_2d)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self._vehicle.apply_control(control)
        self._world.tick()

        obs = self._get_observation()
        info = {
            "collision": self._collision_flag,
            "lane_invasion": self._lane_invasion_flag,
        }
        return obs, info

    def close(self):
        self._cleanup_actors()
        if self._world is not None:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self._world.apply_settings(settings)

    def _get_observation(self):
        import cv2

        raw = self._image_queue.get(timeout=5.0)
        array = np.frombuffer(raw.raw_data, dtype=np.uint8)
        array = array.reshape((raw.height, raw.width, 4))
        rgb = array[:, :, :3][:, :, ::-1]
        resized = cv2.resize(rgb, (self.image_model_size, self.image_model_size))
        image = (resized.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)

        velocity = self._vehicle.get_velocity()
        speed_mps = float(np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
        control = self._vehicle.get_control()
        steer = float(control.steer)

        bearing = 0.0
        if self._goal_location is not None:
            loc = self._vehicle.get_location()
            fwd = self._vehicle.get_transform().get_forward_vector()
            dx = self._goal_location.x - loc.x
            dy = self._goal_location.y - loc.y
            target_yaw = np.arctan2(dy, dx)
            vehicle_yaw = np.arctan2(fwd.y, fwd.x)
            bearing = float(
                np.arctan2(np.sin(target_yaw - vehicle_yaw), np.cos(target_yaw - vehicle_yaw))
            )

        state = np.array([speed_mps, steer, bearing], dtype=np.float32)
        return image, state

    def set_goal(self, location):
        self._goal_location = location

    def _on_image(self, image):
        if not self._image_queue.full():
            self._image_queue.put_nowait(image)
        else:
            try:
                self._image_queue.get_nowait()
            except queue.Empty:
                pass
            self._image_queue.put_nowait(image)

    def _on_collision(self, event):
        self._collision_flag = True

    def _on_lane_invasion(self, event):
        self._lane_invasion_flag = True

    def _cleanup_actors(self):
        for actor in [
            self._chase_camera,
            self._camera,
            self._collision_sensor,
            self._lane_sensor,
            self._vehicle,
        ]:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self._camera = None
        self._chase_camera = None
        self._collision_sensor = None
        self._lane_sensor = None
        self._vehicle = None
        while not self._image_queue.empty():
            try:
                self._image_queue.get_nowait()
            except queue.Empty:
                break
