#!/usr/bin/python

"""
Drive CARLA manual control.
"""

import argparse
import datetime
import logging
import math
import os
import random
import re
import sys

import carla
import pygame
import redis
from common.util import get_actor_display_name
from maneuvers import JTurnManeuver
from pygame.locals import (
    K_0,
    K_9,
    K_BACKQUOTE,
    K_BACKSPACE,
    K_COMMA,
    K_DOWN,
    K_EQUALS,
    K_ESCAPE,
    K_F1,
    K_LEFT,
    K_MINUS,
    K_PERIOD,
    K_RIGHT,
    K_SLASH,
    K_SPACE,
    K_TAB,
    K_UP,
    KMOD_CTRL,
    KMOD_SHIFT,
    K_a,
    K_b,
    K_c,
    K_d,
    K_f,
    K_g,
    K_h,
    K_i,
    K_j,
    K_l,
    K_m,
    K_n,
    K_o,
    K_p,
    K_q,
    K_r,
    K_s,
    K_t,
    K_v,
    K_w,
    K_x,
    K_z,
)
from sensors import (
    CameraManager,
    CollisionSensor,
    GnssSensor,
    IMUSensor,
    LaneInvasionSensor,
    RadarSensor,
)


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x):
        return ' '.join(m.group(0) for m in rgx.finditer(x))

    presets = [
        x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)
    ]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)
    valid_charger_ids = ['vehicle.dodge.charger_2020']
    bps = [bp for bp in bps if bp.id in valid_charger_ids]

    if generation.lower() == 'all':
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    message = 'Actor Generation is not valid. No actor will be spawned.'

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [
                x
                for x in bps
                if int(x.get_attribute('generation')) == int_generation
            ]
            return bps
        else:
            logging.warning(message)
            return []
    except:
        logging.warning(message)
        return []


class World(object):
    def __init__(self, carla_world, hud, args):
        args.filter = 'vehicle.dodge.charger_2020'  # Force filter
        # args.generation = '2'  # Force generation
        args.rolename = 'charger'  # Set appropriate role name

        # Existing initialization code...
        # blueprint = random.choice(
        #     get_actor_blueprints(self.world, self._actor_filter,
        #                        self._actor_generation))

        # Ensure Charger-specific attributes
        # blueprint.set_attribute('driver_id', '1')  # Human driver
        # if blueprint.has_attribute('color'):
        #     colors = ['0.0,0.0,0.0', '0.65,0.17,0.17']  # Black, Dodge Red
        #     blueprint.set_attribute('color', random.choice(colors))
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        self.mode = args.mode
        self.redis_client = None
        self.command_queue = args.command_queue
        if self.mode == 'online':
            try:
                self.redis_client = redis.Redis(
                    host=args.redis_host,
                    port=args.redis_port,
                    db=args.redis_db,
                )
                self.redis_client.ping()
                logging.info(
                    f'World connected to Redis at {args.redis_host}:{args.redis_port}, DB: {args.redis_db}'
                )
            except redis.exceptions.ConnectionError as e:
                logging.error(f'World failed to connect to Redis: {e}')
                sys.exit(1)
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print(f'RuntimeError: {error}')
            print('The server could not send the OpenDRIVE (.xodr) file:')
            print("Ensure it exists, matches the town's name, and is correct.")
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._spawn_point_index = args.spawn
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.iterations = args.iterations
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All,
        ]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = (
            self.camera_manager.index if self.camera_manager is not None else 0
        )
        cam_pos_index = (
            self.camera_manager.transform_index
            if self.camera_manager is not None
            else 0
        )
        # Get a random blueprint.
        blueprint = random.choice(
            get_actor_blueprints(
                self.world, self._actor_filter, self._actor_generation
            )
        )
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values
            )
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values
            )
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(
                blueprint.get_attribute('speed').recommended_values[1]
            )
            self.player_max_speed_fast = float(
                blueprint.get_attribute('speed').recommended_values[2]
            )

        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            print('There are no spawn points available in your map/town.')
            print('Please add some Vehicle Spawn Point to your UE4 scene.')
            sys.exit(1)

        try:
            spawn_point = spawn_points[self._spawn_point_index]
        except Exception:
            spawn_point = None

        # Spawn the player
        if self.player is not None:
            if not spawn_point:
                # Drop vehicle above the ground
                spawn_point = self.player.get_transform()
                spawn_point.location.z += 2.0
                spawn_point.rotation.roll = 0.0
                spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not spawn_point:
                spawn_point = (
                    random.choice(spawn_points)
                    if spawn_points
                    else carla.Transform()
                )
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification(f'Weather: {preset[1]}')
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification(f'LayerMap selected: {selected}')

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification(f'Unloading map layer: {selected}')
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification(f'Loading map layer: {selected}')
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,
        ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world, start_in_autopilot, mode, args):
        self.mode = mode  # 'online' or 'offline'
        # For online mode, use the Redis queue that the orchestrator writes the validated maneuver parameters into.
        if mode == 'online':
            # Instead of using args.command_queue (a separate queue), we use args.redis_queue.
            self.command_queue = args.redis_queue
        else:
            self.command_queue = None  # Not used in offline mode.
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError('Actor type not supported')
        self._steer_cache = 0.0
        self.maneuver = None
        self.iter_count = 0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode, redis_client):
        try:
            # If a maneuver is active, update it
            if self.maneuver is not None:
                if (
                    not hasattr(world.player, 'is_alive')
                    or not world.player.is_alive
                ):
                    logging.error('Vehicle is not valid - resetting maneuver')
                    self.maneuver = None
                    return False

                try:
                    # Get vehicle state before update
                    pre_pos = world.player.get_transform().location

                    ongoing = self.maneuver.update(pygame.time.get_ticks())

                    # Get vehicle state after update
                    post_pos = world.player.get_transform().location

                    # Debug movement
                    if pre_pos != post_pos:
                        logging.info(f'Vehicle moved: {pre_pos} -> {post_pos}')
                    else:
                        logging.warning(
                            'Vehicle position unchanged after update'
                        )

                    if not ongoing:
                        if self.maneuver.save:
                            self.maneuver.save_trajectory()
                        self.maneuver = None
                        world.hud.notification('J-turn completed')
                        world.restart()
                        if self.mode == 'online' and redis_client:
                            self.check_and_start_maneuver(world, redis_client)
                except Exception as e:
                    logging.error(f'Error during maneuver update: {e}')
                    import traceback

                    logging.error(traceback.format_exc())
                    self.maneuver = None

            # If no maneuver is active and in online mode, check queue
            elif self.mode == 'online' and redis_client:
                self.check_and_start_maneuver(world, redis_client)

            # Process other events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                elif event.type == pygame.KEYUP:
                    if self._is_quit_shortcut(event.key):
                        return True
                    elif event.key == K_BACKSPACE:
                        if self._autopilot_enabled:
                            world.player.set_autopilot(False)
                            world.restart()
                            world.player.set_autopilot(True)
                        else:
                            world.restart()
                    elif event.key == K_F1:
                        world.hud.toggle_info()
                    elif (
                        event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT
                    ):
                        world.next_map_layer(reverse=True)
                    elif event.key == K_v:
                        world.next_map_layer()
                    elif (
                        event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT
                    ):
                        world.load_map_layer(unload=True)
                    elif event.key == K_b:
                        world.load_map_layer()
                    elif event.key == K_h or (
                        event.key == K_SLASH
                        and pygame.key.get_mods() & KMOD_SHIFT
                    ):
                        world.hud.help.toggle()
                    elif event.key == K_TAB:
                        world.camera_manager.toggle_camera()
                    elif (
                        event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT
                    ):
                        world.next_weather(reverse=True)
                    elif event.key == K_c:
                        world.next_weather()
                    elif event.key == K_g:
                        world.toggle_radar()
                    elif event.key == K_BACKQUOTE:
                        world.camera_manager.next_sensor()
                    elif event.key == K_n:
                        world.camera_manager.next_sensor()
                    elif event.key == K_w and (
                        pygame.key.get_mods() & KMOD_CTRL
                    ):
                        if world.constant_velocity_enabled:
                            world.player.disable_constant_velocity()
                            world.constant_velocity_enabled = False
                            world.hud.notification(
                                'Disabled Constant Velocity Mode'
                            )
                        else:
                            world.player.enable_constant_velocity(
                                carla.Vector3D(17, 0, 0)
                            )
                            world.constant_velocity_enabled = True
                            world.hud.notification(
                                'Enabled Constant Velocity Mode at 60 km/h'
                            )
                    elif event.key == K_o:
                        try:
                            if world.doors_are_open:
                                world.hud.notification('Closing Doors')
                                world.doors_are_open = False
                                world.player.close_door(carla.VehicleDoor.All)
                            else:
                                world.hud.notification('Opening doors')
                                world.doors_are_open = True
                                world.player.open_door(carla.VehicleDoor.All)
                        except Exception:
                            pass
                    elif event.key == K_t:
                        if world.show_vehicle_telemetry:
                            world.player.show_debug_telemetry(False)
                            world.show_vehicle_telemetry = False
                            world.hud.notification(
                                'Disabled Vehicle Telemetry'
                            )
                        else:
                            try:
                                world.player.show_debug_telemetry(True)
                                world.show_vehicle_telemetry = True
                                world.hud.notification(
                                    'Enabled Vehicle Telemetry'
                                )
                            except Exception:
                                pass
                    elif K_0 < event.key <= K_9:
                        index_ctrl = 0
                        if pygame.key.get_mods() & KMOD_CTRL:
                            index_ctrl = 9
                        world.camera_manager.set_sensor(
                            event.key - 1 - K_0 + index_ctrl
                        )
                    elif event.key == K_r and not (
                        pygame.key.get_mods() & KMOD_CTRL
                    ):
                        world.camera_manager.toggle_recording()
                    elif event.key == K_r and (
                        pygame.key.get_mods() & KMOD_CTRL
                    ):
                        if world.recording_enabled:
                            client.stop_recorder()
                            world.recording_enabled = False
                            world.hud.notification('Recorder is OFF')
                        else:
                            client.start_recorder('manual_recording.rec')
                            world.recording_enabled = True
                            world.hud.notification('Recorder is ON')
                    elif event.key == K_p and (
                        pygame.key.get_mods() & KMOD_CTRL
                    ):
                        # stop recorder
                        client.stop_recorder()
                        world.recording_enabled = False
                        # work around to fix camera at start of replaying
                        current_index = world.camera_manager.index
                        world.destroy_sensors()
                        # disable autopilot
                        self._autopilot_enabled = False
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            "Replaying file 'manual_recording.rec'"
                        )
                        # replayer
                        client.replay_file(
                            'manual_recording.rec', world.recording_start, 0, 0
                        )
                        world.camera_manager.set_sensor(current_index)
                    elif event.key == K_MINUS and (
                        pygame.key.get_mods() & KMOD_CTRL
                    ):
                        if pygame.key.get_mods() & KMOD_SHIFT:
                            world.recording_start -= 10
                        else:
                            world.recording_start -= 1
                        world.hud.notification(
                            f'Recording start time is {world.recording_start}'
                        )
                    elif event.key == K_EQUALS and (
                        pygame.key.get_mods() & KMOD_CTRL
                    ):
                        if pygame.key.get_mods() & KMOD_SHIFT:
                            world.recording_start += 10
                        else:
                            world.recording_start += 1
                        world.hud.notification(
                            f'Recording start time is {world.recording_start}'
                        )
                    if isinstance(self._control, carla.VehicleControl):
                        if event.key == K_f:
                            # Toggle ackermann controller
                            self._ackermann_enabled = (
                                not self._ackermann_enabled
                            )
                            world.hud.show_ackermann_info(
                                self._ackermann_enabled
                            )
                            status = (
                                'Enabled'
                                if self._ackermann_enabled
                                else 'Disabled'
                            )
                            world.hud.notification(
                                f'Ackermann Controller {status}'
                            )
                        if event.key == K_q:
                            if not self._ackermann_enabled:
                                self._control.gear = (
                                    1 if self._control.reverse else -1
                                )
                            else:
                                self._ackermann_reverse *= -1
                                # Reset ackermann control
                                self._ackermann_control = (
                                    carla.VehicleAckermannControl()
                                )
                        elif event.key == K_m:
                            self._control.manual_gear_shift = (
                                not self._control.manual_gear_shift
                            )
                            self._control.gear = (
                                world.player.get_control().gear
                            )
                            if self._control.manual_gear_shift:
                                world.hud.notification('Manual Transmission')
                            else:
                                world.hud.notification(
                                    'Automatic Transmission'
                                )
                        elif (
                            self._control.manual_gear_shift
                            and event.key == K_COMMA
                        ):
                            self._control.gear = max(
                                -1, self._control.gear - 1
                            )
                        elif (
                            self._control.manual_gear_shift
                            and event.key == K_PERIOD
                        ):
                            self._control.gear = self._control.gear + 1
                        elif (
                            event.key == K_p
                            and not pygame.key.get_mods() & KMOD_CTRL
                        ):
                            if not self._autopilot_enabled and not sync_mode:
                                print(
                                    'WARNING: You are currently in asynchronous mode and could '
                                    'experience some issues with the traffic simulation'
                                )
                            self._autopilot_enabled = (
                                not self._autopilot_enabled
                            )
                            world.player.set_autopilot(self._autopilot_enabled)
                            autopilot_status = (
                                'On' if self._autopilot_enabled else 'Off'
                            )
                            world.hud.notification(
                                f'Autopilot {autopilot_status}'
                            )
                        elif (
                            event.key == K_l
                            and pygame.key.get_mods() & KMOD_CTRL
                        ):
                            current_lights ^= carla.VehicleLightState.Special1
                        elif (
                            event.key == K_l
                            and pygame.key.get_mods() & KMOD_SHIFT
                        ):
                            current_lights ^= carla.VehicleLightState.HighBeam
                        elif event.key == K_l:
                            # Use 'L' key to switch between lights:
                            # closed -> position -> low beam -> fog
                            if (
                                not self._lights
                                & carla.VehicleLightState.Position
                            ):
                                world.hud.notification('Position lights')
                                current_lights |= (
                                    carla.VehicleLightState.Position
                                )
                            else:
                                world.hud.notification('Low beam lights')
                                current_lights |= (
                                    carla.VehicleLightState.LowBeam
                                )
                            if self._lights & carla.VehicleLightState.LowBeam:
                                world.hud.notification('Fog lights')
                                current_lights |= carla.VehicleLightState.Fog
                            if self._lights & carla.VehicleLightState.Fog:
                                world.hud.notification('Lights off')
                                current_lights ^= (
                                    carla.VehicleLightState.Position
                                )
                                current_lights ^= (
                                    carla.VehicleLightState.LowBeam
                                )
                                current_lights ^= carla.VehicleLightState.Fog
                        elif event.key == K_i:
                            current_lights ^= carla.VehicleLightState.Interior
                        elif event.key == K_z:
                            current_lights ^= (
                                carla.VehicleLightState.LeftBlinker
                            )
                        elif event.key == K_x:
                            current_lights ^= (
                                carla.VehicleLightState.RightBlinker
                            )
                        elif event.key == K_j:
                            if self.mode == 'online' and redis_client:
                                try:
                                    result = redis_client.blpop(
                                        self.command_queue, timeout=5
                                    )
                                    if result is None:
                                        world.hud.notification(
                                            'No maneuver parameters available'
                                        )
                                    else:
                                        _, maneuver_json = result
                                        from config.config import (
                                            ManeuverParameters,
                                        )

                                        params = ManeuverParameters.model_validate_json(
                                            maneuver_json.decode('utf8')
                                        )
                                        self.maneuver = JTurnManeuver(
                                            world,
                                            maneuver_parameters=params,
                                            base_dir='./logs',
                                            save=True,
                                            online=True,
                                        )
                                        self.maneuver.start(
                                            pygame.time.get_ticks()
                                        )
                                        world.hud.notification(
                                            'Starting Maneuver with online parameters'
                                        )
                                        logging.info(
                                            'J-Turn maneuver started with online parameters'
                                        )
                                except Exception as e:
                                    world.hud.notification(
                                        'Failed to fetch online maneuver parameters'
                                    )
                                    logging.error(
                                        f'Error fetching online maneuver parameters: {e}'
                                    )
                        # Update vehicle lights only if _control is a VehicleControl.
                        if isinstance(self._control, carla.VehicleControl):
                            if self._control.brake:
                                current_lights |= carla.VehicleLightState.Brake
                            else:
                                current_lights &= (
                                    ~carla.VehicleLightState.Brake
                                )
                            if self._control.reverse:
                                current_lights |= (
                                    carla.VehicleLightState.Reverse
                                )
                            else:
                                current_lights &= (
                                    ~carla.VehicleLightState.Reverse
                                )
                            if (
                                current_lights != self._lights
                            ):  # Update only if changed.
                                self._lights = current_lights
                                world.player.set_light_state(
                                    carla.VehicleLightState(self._lights)
                                )
        except Exception as e:
            logging.error(f'Critical error in parse_events: {e}')
            import traceback

            logging.error(traceback.format_exc())
            return True  # Return True to exit the game loop on critical errors

        return False

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(
                    self._control.throttle + 0.01, 1.00
                )
            else:
                self._ackermann_control.speed += (
                    round(milliseconds * 0.005, 2) * self._ackermann_reverse
                )
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= (
                    min(
                        abs(self._ackermann_control.speed),
                        round(milliseconds * 0.005, 2),
                    )
                    * self._ackermann_reverse
                )
                self._ackermann_control.speed = (
                    max(0, abs(self._ackermann_control.speed))
                    * self._ackermann_reverse
                )
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = 0.01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = 0.01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = (
                world.player_max_speed_fast
                if pygame.key.get_mods() & KMOD_SHIFT
                else world.player_max_speed
            )
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (
            key == K_q and pygame.key.get_mods() & KMOD_CTRL
        )

    def check_and_start_maneuver(self, world, redis_client):
        """Helper method to check queue and start maneuver"""
        try:
            result = redis_client.blpop(self.command_queue, timeout=1)
            if result:
                _, maneuver_json = result
                from config.config import ManeuverParameters

                params = ManeuverParameters.model_validate_json(
                    maneuver_json.decode('utf8')
                )
                logging.info(f'Received parameters: {params}')

                self.maneuver = JTurnManeuver(
                    world,
                    maneuver_parameters=params,
                    base_dir='./logs',
                    save=True,
                    online=True,
                )
                self.maneuver.start(pygame.time.get_ticks())
                world.hud.notification(
                    'Starting Maneuver with online parameters'
                )
                logging.info('J-Turn maneuver started with online parameters')
        except Exception as e:
            logging.error(f'Error starting maneuver: {e}')
            import traceback

            logging.error(traceback.format_exc())


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        # Extract accelerometer/gyroscope for formatting
        ax, ay, az = world.imu_sensor.accelerometer
        gx, gy, gz = world.imu_sensor.gyroscope

        self._info_text = [
            f'Server:  {self.server_fps:16.0f} FPS',
            f'Client:  {clock.get_fps():16.0f} FPS',
            '',
            f'Vehicle: {get_actor_display_name(world.player, truncate=20):20}',
            f'Map:     {world.map.name.split("/")[-1]:20}',
            f'Simulation time: {str(datetime.timedelta(seconds=int(self.simulation_time))):12}',
            '',
            f'Speed:   {3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2):15.0f} km/h',
            f'Compass:{compass:17.0f}\N{DEGREE SIGN} {heading:2}',
            f'Accelero: ({ax:5.1f},{ay:5.1f},{az:5.1f})',
            f'Gyroscop: ({gx:5.1f},{gy:5.1f},{gz:5.1f})',
            f'Location:{f"({t.location.x:5.1f}, {t.location.y:5.1f})":>20}',
            f'GNSS:{f"({world.gnss_sensor.lat:2.6f}, {world.gnss_sensor.lon:3.6f})":>24}',
            f'Height:  {t.location.z:18.0f} m',
            '',
        ]
        if isinstance(c, carla.VehicleControl):
            gear_text = {-1: 'R', 0: 'N'}.get(c.gear, c.gear)
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                f'Gear:        {gear_text}',
            ]
            if self._show_ackermann_info:
                self._info_text += [
                    '',
                    'Ackermann Controller:',
                    f'  Target speed: {self._ackermann_control.speed * 3.6:8.0f} km/h',
                ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump),
            ]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            f'Number of vehicles: {len(vehicles):8d}',
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

            def distance(l):
                return math.sqrt(
                    (l.x - t.location.x) ** 2
                    + (l.y - t.location.y) ** 2
                    + (l.z - t.location.z) ** 2
                )

            vehicles = [
                (distance(x.get_location()), x)
                for x in vehicles
                if x.id != world.player.id
            ]
            for d, vehicle in sorted(
                vehicles, key=lambda vehicles: vehicles[0]
            ):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append(f'{int(d):4d}m {vehicle_type}')

    def show_ackermann_info(self, enabled):
        self._show_ackermann_info = enabled

    def update_ackermann_control(self, ackermann_control):
        self._ackermann_control = ackermann_control

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text(f'Error: {text}', (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [
                            (x + 8, v_offset + 8 + (1.0 - y) * 30)
                            for x, y in enumerate(item)
                        ]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2
                        )
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6)
                        )
                        pygame.draw.rect(
                            display, (255, 255, 255), rect, 0 if item[1] else 1
                        )
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6)
                        )
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1
                        )
                        f_val = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (
                                    bar_h_offset + f_val * (bar_width - 6),
                                    v_offset + 8,
                                ),
                                (6, 6),
                            )
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8),
                                (f_val * bar_width, 6),
                            )
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255)
                    )
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (
            0.5 * width - 0.5 * self.dim[0],
            0.5 * height - 0.5 * self.dim[1],
        )
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None
    redis_client = None  # Initialize Redis client variable

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        sim_world = client.get_world()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print(
                'WARNING: You are currently in asynchronous mode and could '
                'experience some issues with the traffic simulation'
            )

        display = pygame.display.set_mode(
            (args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot, args.mode, args)

        # Initialize Redis client if in online mode
        if args.mode == 'online':
            try:
                redis_client = redis.Redis(
                    host=args.redis_host,
                    port=args.redis_port,
                    db=args.redis_db,
                )
                redis_client.ping()
                logging.info(
                    f'Connected to Redis at {args.redis_host}:{args.redis_port}, DB: {args.redis_db}'
                )
            except redis.exceptions.ConnectionError as e:
                logging.error(f'Failed to connect to Redis: {e}')
                sys.exit(1)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(
                client, world, clock, args.sync, redis_client
            ):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if original_settings:
            sim_world.apply_settings(original_settings)

        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client'
    )
    argparser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information',
    )
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)',
    )
    argparser.add_argument(
        '-p',
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)',
    )
    argparser.add_argument(
        '-a', '--autopilot', action='store_true', help='enable autopilot'
    )
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)',
    )
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.dodge.charger_2020',  # Changed from 'vehicle.*'
        help='actor filter (default: "vehicle.dodge.charger_2020")',
    )
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',  # Ensure this matches Charger's generation
        help='restrict to certain actor generation (default: "2")',
    )
    argparser.add_argument(
        '--spawn',
        metavar='SPAWN_POINT',
        default=100,
        type=int,
        help='spawn point index (default: 100)',
    )
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")',
    )
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)',
    )
    argparser.add_argument(
        '--iterations',
        default=1,
        type=int,
        help='Number of maneuver iterations to perform',
    )
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution',
    )
    argparser.add_argument(
        '--mode',
        choices=['offline', 'online'],
        default='offline',
        help='Operation mode: "offline" or "online" (default: "offline")',
    )
    argparser.add_argument(
        '--redis_host',
        default='localhost',
        help='Redis server host (default: localhost)',
    )
    argparser.add_argument(
        '--redis_port',
        type=int,
        default=6379,
        help='Redis server port (default: 6379)',
    )
    argparser.add_argument(
        '--redis_db',
        type=int,
        default=0,
        help='Redis database number (default: 0)',
    )
    argparser.add_argument(
        '--redis_queue',
        default='command_queue',
        help='Redis queue name (default: command_queue)',
    )

    args = argparser.parse_args()

    # Assign redis_queue to command_queue so that the rest of the code stays unchanged.
    args.command_queue = args.redis_queue

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info(f'Listening to server {args.host}:{args.port}')

    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        logging.info('Exiting by user request.')


if __name__ == '__main__':
    main()
