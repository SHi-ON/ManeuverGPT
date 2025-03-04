# Mustang's Lightning Blue color code: #1F43B0
import csv
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import List

from openai import OpenAI
from pydantic import BaseModel

from controller import Keyboard, GPS, Gyro, Supervisor
from vehicle import Driver

llm = OpenAI()

TIME_STEP = 64  # time step for the simulation (64ms is typical)

MAX_SPEED = 80  # maximum speed in km/h
ACCELERATION_STEP = 5.0  # acceleration step for increasing/decreasing speed

MAX_TURN_ANGLE = 0.5  # maximum steering angle in radians (~30 degrees)
TURN_ANGLE_STEP = 0.05  # step for steering angle adjustment

ROLL_THRESHOLD = 1.5  # ~45 degrees in radians
VELOCITY_THRESHOLD = 0.5  # Arbitrary low value to detect stopping

MANUAL_STEERING = False

MODEL_NAME = 'gpt-4o-mini'
# Cost per 1,000 input and output tokens
# https://openai.com/api/pricing/
PROMPT_COST_PER_1K = 0.000_150
COMPLETION_COST_PER_1K = 0.000_600

MANEUVER_NAME = 'Side Wheelie'

SYSTEM_PROMPT = (
    "You are controlling a car in a Webots simulation. "
    "The car must be driven safely, avoiding excessive tilting or leaving the ground. "
    "The car's parameters include:\n"
    "- Steering angle range: -0.75 to 0.75 radians\n"
    "- Speed range: -80 to 80 km/h (negative values indicate braking and reverse)\n"
    f"Your objective is to steer the car and adjust the speed to achieve a controlled {MANEUVER_NAME} maneuver. "
    "Generate a series of steering and speed instructions to keep the car balanced and stable.\n"
    "Parameters:\n"
    "- cruising_speed: the current speed of the car\n"
    "- steering_angle: the car's steering angle\n"
    "- roll_velocity: the car’s angular velocity around the x-axis\n"
    "Output a list of the next steps in JSON format with keys 'steering_angle' and 'cruising_speed' for each step."
)

# Initialize the Robot instance, Driver, and Keyboard
supervisor = Supervisor()
driver = Driver()

# Initialize the Keyboard controller, GPS, and Gyro devices
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

gps: GPS = supervisor.getDevice('gps')
gps.enable(TIME_STEP)

gyro: Gyro = supervisor.getDevice('gyro')
gyro.enable(TIME_STEP)

# Set initial cruising speed and steering angle
cruising_speed = 25
steering_angle = 0.0


class Action(BaseModel):
    steering_angle: float
    cruising_speed: float


class ActionInstructions(BaseModel):
    actions: List[Action]


class LLMQ(threading.Thread):
    def __init__(self, queue, llm, system_prompt):
        # Set daemon=True so it exits when main thread exits
        super().__init__(daemon=True)
        self.queue = queue
        self.llm = llm
        self.system_prompt = system_prompt
        self._running = True
        self.current_state = None

    def update_state(self, cruising_speed, steering_angle, roll_velocity):
        self.current_state = {
            'cruising_speed': cruising_speed,
            'steering_angle': steering_angle,
            'roll_velocity': roll_velocity
        }

    def run(self, num_steps=50):
        while self.running:
            if not self.current_state:
                time.sleep(0.1)
                continue
            try:
                response = self.llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            'role': 'system',
                            'content': self.system_prompt
                        },
                        {
                            'role': 'user',
                            'content': (
                                f'cruising_speed: {self.current_state["cruising_speed"]}, '
                                f'steering_angle: {self.current_state["steering_angle"]}, '
                                f'roll_velocity: {self.current_state["roll_velocity"]}, '
                                f'Generate the next {num_steps} steps to achieve the desired maneuver safely.'
                            )
                        }
                    ],
                    response_format={
                        'type': 'json_schema',
                        'json_schema': {"name": "whocares",
                                        "schema": ActionInstructions.model_json_schema()}
                    },
                )
                self.log_usage_cost(response.usage)

                content = response.choices[0].message.content
                json_parsed = json.loads(content)
                action_instructions = ActionInstructions(**json_parsed)
                # Enqueue the generated action instructions
                self.queue.put(action_instructions)
            except Exception as e:
                logging.error(f"Error in LLM generation: {e}")
            time.sleep(0.1)  # Small delay to prevent CPU overuse

    @property
    def running(self):
        return self._running

    @running.setter
    def running(self, value):
        if not isinstance(value, bool):
            raise ValueError('running must be a bool value')
        self._running = value

    def stop(self):
        self._running = False

    def log_usage_cost(self, usage):
        """Log the usage cost of the LLM response to a CSV file."""
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        prompt_cost = (prompt_tokens / 1000) * PROMPT_COST_PER_1K
        completion_cost = (completion_tokens / 1000) * COMPLETION_COST_PER_1K
        total_cost = prompt_cost + completion_cost

        print('Details LLM of usage:', usage.prompt_tokens_details)

        # Prepare the row data
        timestamp = datetime.utcnow().isoformat()
        row_data = [
            timestamp,
            MODEL_NAME,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            prompt_cost,
            completion_cost,
            total_cost
        ]

        # CSV file path
        csv_path = 'llm_usage_costs.csv'

        # Check if file exists to write headers
        file_exists = os.path.isfile(csv_path)

        # Write to CSV file
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write headers if file is new
            if not file_exists:
                writer.writerow([
                    'Timestamp',
                    'Model',
                    'Prompt Tokens',
                    'Completion Tokens',
                    'Total Tokens',
                    'Prompt Cost ($)',
                    'Completion Cost ($)',
                    'Total Cost ($)'
                ])

            writer.writerow(row_data)


def reset_environment():
    logging.warning('Resetting the 🚀 simulation environment...')
    # Reset the simulation environment
    supervisor.simulationResetPhysics()
    supervisor.simulationReset()
    driver.simulationResetPhysics()
    driver.simulationReset()

    driver.setCruisingSpeed(0)
    driver.setSteeringAngle(0)
    gps.enable(TIME_STEP)
    gyro.enable(TIME_STEP)


def main():
    global cruising_speed, steering_angle

    llmq = None
    current_action_instructions = None
    action_index = 0

    if not MANUAL_STEERING:
        # Create a queue for communication between threads
        llm_queue = queue.Queue()
        # Create and start the GPT generator thread
        llmq = LLMQ(llm_queue, llm, SYSTEM_PROMPT)
        llmq.start()

    print('mode:', supervisor.simulationGetMode())

    # Main simulation loop
    while supervisor.step(TIME_STEP) != -1:

        # Detect simulation reset
        current_mode = supervisor.simulationGetMode()
        if current_mode == 0:
            llmq.running = False
        else:
            llmq.running = True

        # Check keyboard inputs
        key = keyboard.getKey()

        # Speed control with UP and DOWN arrow keys
        if key == Keyboard.UP:
            cruising_speed = min(cruising_speed + ACCELERATION_STEP, MAX_SPEED)
        elif key == Keyboard.DOWN:
            cruising_speed = max(cruising_speed - ACCELERATION_STEP,
                                 -MAX_SPEED)
        # Steering control with LEFT and RIGHT arrow keys
        elif key == Keyboard.LEFT:
            steering_angle = max(steering_angle - TURN_ANGLE_STEP,
                                 -MAX_TURN_ANGLE)
        elif key == Keyboard.RIGHT:
            steering_angle = min(steering_angle + TURN_ANGLE_STEP,
                                 MAX_TURN_ANGLE)
        else:
            # Gradually reduce the steering angle back to center
            if steering_angle < 0:
                steering_angle = min(steering_angle + TURN_ANGLE_STEP, 0)
            elif steering_angle > 0:
                steering_angle = max(steering_angle - TURN_ANGLE_STEP, 0)

        gps_position = gps.getValues()
        gps_linear_velocity = gps.getSpeedVector()
        gyro_angular_velocity = gyro.getValues()

        # Check for excessive roll (tilt)
        # Angular velocity around x-axis
        roll_velocity = gyro_angular_velocity[0]
        if abs(roll_velocity) > ROLL_THRESHOLD:
            logging.warning("Failure detected: Excessive roll (tilt)")
            reset_environment()

        # Check for abnormal position change
        # Car lifted too high, indicating potential rollover
        if gps_position[2] > 0.5:
            logging.warning("Failure detected: Car is lifted off the ground")
            reset_environment()

        if not MANUAL_STEERING:
            # Update GPT generator with current state
            llmq.update_state(cruising_speed, steering_angle, roll_velocity)
            # Check for new action instructions

            if current_action_instructions and \
                    action_index < len(current_action_instructions.actions):
                steering_step = current_action_instructions.actions[
                    action_index]
                action_index += 1
            else:
                try:
                    current_action_instructions = llm_queue.get_nowait()
                    action_index = 0
                    steering_step = current_action_instructions.actions[
                        action_index]
                except queue.Empty:
                    logging.warning('No new action instructions')
                    continue

            cruising_speed = steering_step.cruising_speed
            steering_angle = steering_step.steering_angle

        # Apply the cruising speed and steering angle to the driver
        driver.setCruisingSpeed(cruising_speed)
        driver.setSteeringAngle(steering_angle)

        print(
            f'Cruising Speed: {cruising_speed, driver.getCurrentSpeed()} km/h,'
            f' Steering Angle: {steering_angle} radians,'
            f' ang vel: {gyro_angular_velocity[0]},')

        # supervisor.step(TIME_STEP)

    # End of the main simulation loop
    if not MANUAL_STEERING:
        # Cleanup
        llmq.stop()
        llmq.join()


if __name__ == '__main__':
    if not 'PYCHARM_HOSTED' in os.environ:
        main()
