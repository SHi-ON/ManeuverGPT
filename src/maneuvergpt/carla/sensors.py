import collections
import math
import weakref

import carla
import numpy as np
import pygame
from carla import ColorConverter as cc
from common.util import get_actor_display_name


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith('walker.pedestrian'):
            self._camera_transforms = [
                (
                    carla.Transform(
                        carla.Location(
                            x=-2.0 * bound_x, y=+0.0 * bound_y, z=2.0 * bound_z
                        ),
                        carla.Rotation(pitch=8.0),
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(
                            x=+0.8 * bound_x, y=+0.0 * bound_y, z=1.3 * bound_z
                        )
                    ),
                    Attachment.Rigid,
                ),
                (
                    carla.Transform(
                        carla.Location(
                            x=+1.9 * bound_x, y=+1.0 * bound_y, z=1.2 * bound_z
                        )
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(
                            x=-2.8 * bound_x, y=+0.0 * bound_y, z=4.6 * bound_z
                        ),
                        carla.Rotation(pitch=6.0),
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(
                            x=-1.0, y=-1.0 * bound_y, z=0.4 * bound_z
                        )
                    ),
                    Attachment.Rigid,
                ),
            ]
        else:
            self._camera_transforms = [
                (
                    carla.Transform(
                        carla.Location(x=-2.5, z=0.0),
                        carla.Rotation(pitch=-8.0),
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(carla.Location(x=1.6, z=1.7)),
                    Attachment.Rigid,
                ),
                (
                    carla.Transform(
                        carla.Location(x=2.5, y=0.5, z=0.0),
                        carla.Rotation(pitch=-8.0),
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(x=-4.0, z=2.0),
                        carla.Rotation(pitch=6.0),
                    ),
                    Attachment.SpringArmGhost,
                ),
                (
                    carla.Transform(
                        carla.Location(x=0, y=-2.5, z=-0.0),
                        carla.Rotation(yaw=90.0),
                    ),
                    Attachment.Rigid,
                ),
            ]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            [
                'sensor.camera.depth',
                cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)',
                {},
            ],
            [
                'sensor.camera.semantic_segmentation',
                cc.Raw,
                'Camera Semantic Segmentation (Raw)',
                {},
            ],
            [
                'sensor.camera.semantic_segmentation',
                cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)',
                {},
            ],
            [
                'sensor.camera.instance_segmentation',
                cc.CityScapesPalette,
                'Camera Instance Segmentation (CityScapes Palette)',
                {},
            ],
            [
                'sensor.camera.instance_segmentation',
                cc.Raw,
                'Camera Instance Segmentation (Raw)',
                {},
            ],
            [
                'sensor.lidar.ray_cast',
                None,
                'Lidar (Ray-Cast)',
                {'range': '50'},
            ],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            [
                'sensor.camera.rgb',
                cc.Raw,
                'Camera RGB Distorted',
                {
                    'lens_circle_multiplier': '3.0',
                    'lens_circle_falloff': '3.0',
                    'chromatic_aberration_intensity': '0.5',
                    'chromatic_aberration_offset': '0',
                },
            ],
            ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
            ['sensor.camera.normals', cc.Raw, 'Camera Normals', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(
            self._camera_transforms
        )
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = (
            True
            if self.index is None
            else (
                force_respawn
                or (self.sensors[index][2] != self.sensors[self.index][2])
            )
        )
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][
                    1
                ],
            )
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image)
            )
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification(
            'Recording %s' % ('On' if self.recording else 'Off')
        )

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(
                image.raw_data,
                dtype=np.dtype(
                    [
                        ('x', np.uint16),
                        ('y', np.uint16),
                        ('t', np.int64),
                        ('pol', np.bool),
                    ]
                ),
            )
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[
                dvs_events[:]['y'],
                dvs_events[:]['x'],
                dvs_events[:]['pol'] * 2,
            ] = 255
            self.surface = pygame.surfarray.make_surface(
                dvs_img.swapaxes(0, 1)
            )
        elif self.sensors[self.index][0].startswith(
            'sensor.camera.optical_flow'
        ):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent
        )
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event)
        )

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith('vehicle.'):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find(
                'sensor.other.lane_invasion'
            )
            self.sensor = world.spawn_actor(
                bp, carla.Transform(), attach_to=self._parent
            )
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda event: LaneInvasionSensor._on_invasion(weak_self, event)
            )

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=1.0, z=2.8)),
            attach_to=self._parent,
        )
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(weak_self, event)
        )

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent
        )
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data)
        )

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)),
        )
        self.gyroscope = (
            max(
                limits[0],
                min(limits[1], math.degrees(sensor_data.gyroscope.x)),
            ),
            max(
                limits[0],
                min(limits[1], math.degrees(sensor_data.gyroscope.y)),
            ),
            max(
                limits[0],
                min(limits[1], math.degrees(sensor_data.gyroscope.z)),
            ),
        )
        self.compass = math.degrees(sensor_data.compass)


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5  # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z + 0.05),
                carla.Rotation(pitch=5),
            ),
            attach_to=self._parent,
        )
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(
                weak_self, radar_data
            )
        )

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll,
                ),
            ).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = (
                detect.velocity / self.velocity_range
            )  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(-1.0, 0.0, -1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b),
            )
