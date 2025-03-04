import argparse
import csv
import datetime
import json
import logging
import pathlib
import threading
import time
from queue import Queue, Empty
from typing import Optional, Dict

import carla
import pygame
import redis
from pydantic import BaseModel, ValidationError


# from config.config import ManeuverParameters, PhaseParameters


class PhaseParameters(BaseModel):
    throttle: float
    steering_angle: float
    reverse: bool
    brake: float
    duration: float


class ManeuverParameters(BaseModel):
    phase_1: PhaseParameters
    phase_2: PhaseParameters
    phase_3: PhaseParameters
    phase_4: PhaseParameters
    phase_5: PhaseParameters
    success_conditions: Dict[str, float]


def load_maneuver_from_orchestrator(world) -> Optional[ManeuverParameters]:
    """Direct parameter retrieval from orchestrator"""
    if not hasattr(world, 'orchestrator'):
        logging.warning("No orchestrator found in world")
        return None

    try:
        # Directly access orchestrator's latest validated parameters
        params = world.orchestrator.current_maneuver_params
        if params:
            return ManeuverParameters(**params)
        return None
    except ValidationError as e:
        logging.error(f"Invalid orchestrator parameters: {e}")
        return None


def load_latest_maneuver_parameters(directory: str = None) -> Optional[
    ManeuverParameters]:
    """
    Load the latest maneuver parameters from the controller directory.
    
    Args:
        directory (str, optional): Override directory path if needed.
    
    Returns:
        ManeuverParameters or None if loading fails.
    """

    # Look in the controller root directory
    maneuver_dir = pathlib.Path(__file__).parent.parent
    if directory:
        maneuver_dir = pathlib.Path(directory)
    if not maneuver_dir.exists():
        logging.error(f"Controller directory '{maneuver_dir}' does not exist.")
        return None

    json_files = list(maneuver_dir.glob('validated_maneuver.json'))
    if not json_files:
        logging.error(f"No maneuver JSON files found in '{maneuver_dir}'.")
        return None

    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    logging.info(f"Loading maneuver parameters from '{latest_file}'")

    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        maneuver_params = ManeuverParameters(**data)
        logging.info("Maneuver parameters successfully loaded and validated.")
        return maneuver_params
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed for '{latest_file}': {e}")
    except ValidationError as ve:
        logging.error(f"Pydantic validation error for '{latest_file}': {ve}")
    except Exception as e:
        logging.error(
            f"Unexpected error while loading maneuver parameters: {e}")

    return None


class JTurnManeuver:
    """
    J-turn maneuver for a vehicle.
    Supports both offline and online parameter modes.
    """

    def __init__(
            self,
            world,
            maneuver_parameters: Optional[ManeuverParameters] = None,
            base_dir='./logs',
            save=True,
            online=True  # New Flag to Determine Mode
    ):
        self.world = world
        self.params = maneuver_parameters
        self.base_dir = pathlib.Path(base_dir)
        self.save = save
        self.online = online
        if self.online and self.params is None:
            logging.error(
                "Online Mode: No ManeuverParameters provided externally.")
            raise ValueError("Online Mode requires ManeuverParameters.")
        self.vehicle = world.player
        self.control = carla.VehicleControl()
        self.start_time = None
        self.start_datetime = None
        self.start_transform = None
        self.trajectory = []
        self.phase = 0

    def set_maneuver_parameters(self, maneuver_parameters: ManeuverParameters):
        """
        Allows setting maneuver parameters dynamically in online mode.
        """
        if self.online:
            self.params = maneuver_parameters
            logging.info(
                "Online Mode: ManeuverParameters updated dynamically.")
        else:
            logging.warning(
                "Attempted to set ManeuverParameters in Offline Mode. Operation ignored.")

    def start(self, current_time):
        self.start_time = current_time
        self.start_datetime = datetime.datetime.now()
        self.start_transform = self.vehicle.get_transform()
        self.phase = 1
        logging.info("Starting J-turn")

    def update(self, current_time):
        elapsed = (
                              current_time - self.start_time) / 1000.0  # Convert ms to seconds

        try:
            current_phase_key = f'phase_{self.phase}'
            phase_params: PhaseParameters = getattr(self.params,
                                                    current_phase_key)

            if elapsed < phase_params.duration:
                # Verify vehicle exists and is valid
                if not self.vehicle or not self.vehicle.is_alive:
                    logging.error("Vehicle is not valid!")
                    return False

                # Get current vehicle state for debugging
                current_transform = self.vehicle.get_transform()
                current_velocity = self.vehicle.get_velocity()

                # Set and apply control
                self.control.reverse = phase_params.reverse
                self.control.throttle = phase_params.throttle
                self.control.steer = phase_params.steering_angle
                self.control.brake = phase_params.brake

                # Debug log before applying control
                logging.info(f"""
Phase {self.phase} State:
- Time: {elapsed:.2f}/{phase_params.duration:.2f}s
- Location: ({current_transform.location.x:.2f}, {current_transform.location.y:.2f})
- Velocity: ({current_velocity.x:.2f}, {current_velocity.y:.2f})
- Controls: throttle={self.control.throttle:.2f}, steer={self.control.steer:.2f}, brake={self.control.brake:.2f}, reverse={self.control.reverse}
""")

                try:
                    self.vehicle.apply_control(self.control)
                    # Verify control was applied
                    actual_control = self.vehicle.get_control()
                    logging.info(
                        f"Applied controls verified: throttle={actual_control.throttle:.2f}, steer={actual_control.steer:.2f}")
                except Exception as e:
                    logging.error(f"Failed to apply vehicle control: {e}")
                    return False

                self.track(current_time)
                return True
            else:
                self.phase += 1
                if self.phase > 5:
                    # Reset controls when maneuver completes
                    self.control = carla.VehicleControl()
                    self.vehicle.apply_control(self.control)
                    logging.info("J-turn completed")
                    return False
                else:
                    self.start_time = current_time
                    logging.info(f"Transitioning to Phase {self.phase}")
                    return True

        except AttributeError as ae:
            logging.error(f"Invalid phase key '{current_phase_key}': {ae}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during maneuver execution: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def track(self, current_time):
        # Record vehicle position, orientation, and velocity at each time step
        current_transform = self.vehicle.get_transform()
        current_velocity = self.vehicle.get_velocity()
        current_acceleration = self.vehicle.get_acceleration()
        self.trajectory.append({
            'timestamp': current_time,
            'location': current_transform.location,
            'rotation': current_transform.rotation,
            'velocity': current_velocity,
            'acceleration': current_acceleration,
        })

    def save_trajectory(self):
        if not self.trajectory:
            return

        points = []
        for point in self.trajectory:
            points.append({
                'timestamp': point['timestamp'],
                'x': round(point['location'].x, 4),
                'y': round(point['location'].y, 4),
                'z': round(point['location'].z, 4),
                'pitch': round(point['rotation'].yaw, 4),
                'yaw': round(point['rotation'].pitch, 4),
                'roll': round(point['rotation'].roll, 4),
                'vx': round(point['velocity'].x, 4),
                'vy': round(point['velocity'].y, 4),
                'vz': round(point['velocity'].z, 4),
                'ax': round(point['acceleration'].x, 4),
                'ay': round(point['acceleration'].y, 4),
                'az': round(point['acceleration'].z, 4),
            })

        start_datetime_str = self.start_datetime.strftime('%Y%m%d_%H%M%S')
        file_path = self.base_dir / f'j_turn_{start_datetime_str}.csv'
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as file_handle:
            csv_writer = csv.DictWriter(file_handle,
                                        fieldnames=points[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(points)

    def visualize_trajectory(self):
        # Draw trajectory points
        for point in self.trajectory:
            self.world.debug.draw_point(point['location'],
                                        size=0.1,
                                        color=carla.Color(255, 0, 0),
                                        life_time=20.0)

    def execute_maneuver(self):
        logging.info("Starting J-Turn Maneuver")
        for phase_name, phase in self.params.dict().items():
            if phase_name.startswith('phase'):
                throttle = phase['throttle']
                steering_angle = phase['steering_angle']
                reverse = phase['reverse']
                brake = phase['brake']
                duration = phase['duration']

                logging.info(
                    f"Executing {phase_name}: Throttle={throttle}, Steering Angle={steering_angle}, Reverse={reverse}, Brake={brake}, Duration={duration}s")
                # Apply controls to the vehicle in CARLA
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=steering_angle,
                    brake=brake,
                    reverse=reverse
                )
                self.world.player.apply_control(control)
                time.sleep(duration)  # Wait for the duration of this phase
        logging.info("J-Turn Maneuver Completed")
        if self.save:
            self.save_trajectory()


class YTurnManeuver:
    """
    Y-turn (three-point turn) maneuver for a vehicle.

    Phases:
    0: not started,
    1: forward-right,
    2: reverse-left,
    3: forward-straight,
    4: stabilize,
    5: done
    """

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.control = carla.VehicleControl()
        self.start_time = None
        self.phase = 0

    def start(self, current_time):
        self.start_time = current_time
        self.phase = 1
        logging.info("Starting Y-turn")

    def update(self, current_time):
        elapsed = current_time - self.start_time

        if self.phase == 1:  # Forward-right
            if elapsed < 2000:
                self.control.reverse = False
                self.control.throttle = 0.5
                self.control.steer = 1.0  # Full right turn
                logging.info("Phase 1: Moving forward-right...")
            else:
                self.phase = 2
                self.start_time = current_time

        elif self.phase == 2:  # Reverse-left
            if elapsed < 2000:
                self.control.reverse = True
                self.control.throttle = 0.5
                self.control.steer = -1.0  # Full left turn while reversing
                logging.info("Phase 2: Reversing-left...")
            else:
                self.phase = 3
                self.start_time = current_time

        elif self.phase == 3:  # Forward-straight
            if elapsed < 1500:
                self.control.reverse = False
                self.control.throttle = 0.5
                self.control.steer = 0.3  # Slight right to straighten
                logging.info("Phase 3: Moving forward-straight...")
            else:
                self.phase = 4
                self.start_time = current_time

        elif self.phase == 4:  # Stabilize
            if elapsed < 1000:
                self.control.throttle = 0.3
                self.control.steer = 0.0
                logging.info("Phase 4: Stabilizing...")
            else:
                self.phase = 5

        elif self.phase == 5:  # Done
            self.control = carla.VehicleControl()
            self.phase = 0
            logging.info("Y-turn completed")
            return False  # Indicate that Y-turn is complete

        self.vehicle.apply_control(self.control)
        return True  # Indicate that Y-turn is ongoing


class DonutManeuver:
    """
    Donut maneuver for a vehicle.

    Phases:
    0: not started,
    1: initiate spin,
    2: maintain spin,
    3: exit spin,
    4: stabilize,
    5: done
    """

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.control = carla.VehicleControl()
        self.start_time = None
        self.phase = 0

    def start(self, current_time):
        self.start_time = current_time
        self.phase = 1
        logging.info("Starting Donut maneuver")

    def update(self, current_time):
        elapsed = current_time - self.start_time

        if self.phase == 1:  # Initiate Spin
            if elapsed < 500:
                self.control.throttle = 0.5
                self.control.steer = 1.0
                logging.info("Phase 1: Initiating spin...")
            else:
                self.phase = 2
                self.start_time = current_time  # reset time for next phase

        elif self.phase == 2:  # Maintain Spin
            if elapsed < 3000:
                self.control.throttle = 0.7
                self.control.steer = 1.0
                logging.info("Phase 2: Maintaining spin...")
            else:
                self.phase = 3
                self.start_time = current_time

        elif self.phase == 3:  # Exit Spin
            if elapsed < 500:
                self.control.throttle = 0.5
                self.control.steer = 0.0
                logging.info("Phase 3: Exiting spin...")
            else:
                self.phase = 4
                self.start_time = current_time

        elif self.phase == 4:  # Stabilize
            if elapsed < 1000:
                self.control.throttle = 0.0
                self.control.steer = 0.0
                logging.info("Phase 4: Stabilizing...")
            else:
                self.phase = 5

        elif self.phase == 5:  # Done
            self.control = carla.VehicleControl()
            self.phase = 0
            logging.info("Donut maneuver completed")
            return False  # Indicate that Donut maneuver is complete

        self.vehicle.apply_control(self.control)
        return True  # Indicate that Donut maneuver is ongoing


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Maneuvers Executor for CARLA")
    parser.add_argument(
        '--mode',
        choices=['offline', 'online'],
        default='online',
        help='Operation mode: "offline" or "online" (default: "online")'
    )
    parser.add_argument(
        '--redis_host',
        default='localhost',
        help='Redis server host (default: localhost)'
    )
    parser.add_argument(
        '--redis_port',
        type=int,
        default=6379,
        help='Redis server port (default: 6379)'
    )
    parser.add_argument(
        '--redis_db',
        type=int,
        default=0,
        help='Redis database number (default: 0)'
    )
    parser.add_argument(
        '--maneuver_queue',
        default='maneuver_queue',
        help='Redis maneuver queue name (default: maneuver_queue)'
    )
    parser.add_argument(
        '--base_dir',
        default='./logs',
        help='Directory to save trajectories (default: ./logs)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Enable trajectory saving'
    )
    return parser.parse_args()


def connect_to_redis(host, port, db):
    """Establish a connection to the Redis server."""
    try:
        client = redis.Redis(host=host, port=port, db=db)
        client.ping()
        logging.info(f"Connected to Redis at {host}:{port}, DB: {db}")
        return client
    except redis.exceptions.ConnectionError as e:
        logging.error(f"Failed to connect to Redis: {e}")
        return None


def redis_listener(redis_client, queue_name, local_queue):
    """Listener thread that pushes incoming maneuvers from Redis to a local queue."""
    try:
        while True:
            _, data = redis_client.blpop(queue_name)
            maneuver_json = data.decode('utf-8')
            try:
                maneuver = ManeuverParameters.model_validate_json(
                    maneuver_json)
                logging.debug(
                    f"Received Maneuver Parameters:\n{maneuver.model_dump_json(indent=2)}")
                local_queue.put(maneuver)
            except ValidationError as ve:
                logging.error(f"Pydantic validation error: {ve}")
    except Exception as e:
        logging.error(f"Redis listener encountered an error: {e}")


def execute_maneuver(maneuver_params: ManeuverParameters, base_dir: str,
                     save_data: bool):
    """
    Initializes and executes the JTurnManeuver with the provided parameters.
    """
    # Initialize CARLA world and other necessary components
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Initialize pygame for rendering if needed
    pygame.init()
    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("Carla Simulation")
    clock = pygame.time.Clock()

    try:
        # Initialize JTurnManeuver
        j_turn = JTurnManeuver(
            world=world,
            maneuver_parameters=maneuver_params,
            base_dir=base_dir,
            save=save_data,
            online=True  # Set to True if receiving parameters online
        )

        # Start the maneuver
        current_time = pygame.time.get_ticks()
        j_turn.start(current_time)

        # Execute the maneuver loop
        while j_turn.update(current_time):
            clock.tick_busy_loop(60)
            # Parse pygame events to allow for clean exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    j_turn.save_trajectory()
                    return
            # Advance simulation
            world.tick()
            # Render if needed
            pygame.display.flip()

        # Save trajectory data if enabled
        if j_turn.save:
            j_turn.save_trajectory()

        logging.info("Maneuver execution completed.")

    except Exception as e:
        logging.error(f"Error during maneuver execution: {e}")
    finally:
        # Clean up simulation settings
        sim_world = world.get_settings()
        sim_world.no_rendering_mode = False
        world.apply_settings(sim_world)
        pygame.quit()


def main():
    args = parse_arguments()

    # Initialize logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize Redis connection
    redis_client = connect_to_redis(args.redis_host, args.redis_port,
                                    args.redis_db)
    if not redis_client:
        logging.error("Exiting due to Redis connection failure.")
        return

    # Initialize CARLA client and world
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)  # seconds
        world = client.get_world()
        logging.info("Connected to CARLA server")
    except Exception as e:
        logging.error(f"Failed to connect to CARLA server: {e}")
        return

    # Initialize a local queue for maneuvers
    maneuver_queue = Queue()

    if args.mode == 'online':
        # Start Redis listener thread
        listener_thread = threading.Thread(target=redis_listener, args=(
        redis_client, args.maneuver_queue, maneuver_queue), daemon=True)
        listener_thread.start()
        logging.info(
            f"Started Redis listener thread on queue '{args.maneuver_queue}'")

    try:
        while True:
            if args.mode == 'offline':
                # Load parameters from disk
                try:
                    with open("validated_maneuver.json", "r") as f:
                        params = ManeuverParameters.parse_raw(f.read())
                        logging.info(
                            "Loaded maneuver parameters from 'validated_maneuver.json'")
                        logging.debug(
                            f"Maneuver Parameters:\n{params.model_dump_json(indent=2)}")
                except FileNotFoundError:
                    logging.error(
                        "validated_maneuver.json not found. Ensure orchestrator has generated it.")
                    time.sleep(1)
                    continue
                except ValidationError as ve:
                    logging.error(f"Pydantic validation error: {ve}")
                    time.sleep(1)
                    continue

                # Execute the maneuver
                execute_maneuver(params, args.base_dir, args.save)

            elif args.mode == 'online':
                try:
                    # Wait for a maneuver to be available in the queue
                    params = maneuver_queue.get(
                        timeout=30)  # Adjust timeout as needed
                    logging.info("Starting execution of received maneuver")

                    # Execute the maneuver
                    execute_maneuver(params, args.base_dir, args.save)
                except Empty:
                    logging.warning(
                        "No maneuver received within the timeout period.")
                except ValidationError as ve:
                    logging.error(
                        f"Pydantic validation error during maneuver execution: {ve}")
                except Exception as e:
                    logging.error(f"Error during maneuver execution: {e}")

            # Optional: Implement a mechanism to terminate after certain conditions
            time.sleep(1)  # Example delay between maneuvers

    except KeyboardInterrupt:
        logging.info("Maneuvers interrupted by user.")

    finally:
        if args.mode == 'online' and redis_client:
            redis_client.close()
            logging.info("Redis connection closed.")
        logging.info("Maneuvers shutdown.")


if __name__ == "__main__":
    main()
