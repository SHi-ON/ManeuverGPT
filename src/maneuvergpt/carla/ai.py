# ai.py
import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import redis
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from common.config import ManeuverParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Pydantic models for structured output
class PhaseParameters(BaseModel):
    throttle: float
    steering_angle: float
    reverse: bool = False
    brake: float = 0.0
    duration: float  # Duration in seconds


# class ManeuverParameters(BaseModel):
#     phase_1: PhaseParameters
#     phase_2: PhaseParameters
#     phase_3: PhaseParameters
#     phase_4: PhaseParameters
#     phase_5: PhaseParameters
#     success_conditions: dict = {
#         "max_roll": 0.5,
#         "min_speed": 10.0,
#         "max_tilt": 0.3
#     }

class ManeuverCollection(BaseModel):
    maneuvers: List[ManeuverParameters]


SYSTEM_PROMPT = """
You are an autonomous vehicle maneuver configuration generator tasked with creating precise parameters for a J-Turn maneuver. The maneuver consists of 5 distinct phases, each with specific behaviors to ensure a controlled and safe execution.

### Parameter Ranges and Increments:
- **Throttle**: 0.0 to 1.0 (increments of 0.1)
- **Steering Angle**: -1.0 to 1.0 (increments of 0.2)
- **Brake**: 0.0 to 1.0 (increments of 0.1)
- **Duration**: 0.0 to 5.0 seconds (increments of 0.3 seconds)

### Phases and Requirements:

1. **Phase 1: Reverse**
    - **Reverse**: True
    - **Throttle**: 0.8 to 1.0
    - **Steering Angle**: -0.2 to 0.2
    - **Brake**: 0.0
    - **Duration**: 2.4 to 3.6 seconds

2. **Phase 2: Continue Reversing and Initiate Turn**
    - **Reverse**: True
    - **Throttle**: 0.6 to 0.8
    - **Steering Angle**: 0.8 to 1.0
    - **Brake**: 0.0
    - **Duration**: 0.3 to 0.9 seconds

3. **Phase 3: Counter-steer**
    - **Reverse**: False
    - **Throttle**: 0.8 to 1.0
    - **Steering Angle**: -0.6 to -0.4
    - **Brake**: 0.0
    - **Duration**: 0.3 to 0.6 seconds

4. **Phase 4: Stabilize**
    - **Reverse**: False
    - **Throttle**: 0.4 to 0.6
    - **Steering Angle**: -0.2 to 0.2
    - **Brake**: 0.0
    - **Duration**: 0.9 to 1.5 seconds

5. **Phase 5: Complete Maneuver**
    - **Reverse**: False
    - **Throttle**: 0.0
    - **Steering Angle**: 0.0
    - **Brake**: 0.8 to 1.0
    - **Duration**: 0.0 seconds

### Success Conditions (with ranges):
- **Max Roll**: 0.3 to 0.5 radians (increments of 0.1)
- **Min Speed**: 5.0 to 15.0 km/h (increments of 2.5)
- **Max Tilt**: 0.2 to 0.5 radians (increments of 0.1)

### Required JSON Structure:
{
    "phase_1": {
        "throttle": float,
        "steering_angle": float,
        "reverse": bool,
        "brake": float,
        "duration": float
    },
    "phase_2": {
        "throttle": float,
        "steering_angle": float,
        "reverse": bool,
        "brake": float,
        "duration": float
    },
    "phase_3": {
        "throttle": float,
        "steering_angle": float,
        "reverse": bool,
        "brake": float,
        "duration": float
    },
    "phase_4": {
        "throttle": float,
        "steering_angle": float,
        "reverse": bool,
        "brake": float,
        "duration": float
    },
    "phase_5": {
        "throttle": float,
        "steering_angle": float,
        "reverse": bool,
        "brake": float,
        "duration": float
    },
    "success_conditions": {
        "max_roll": float,
        "min_speed": float,
        "max_tilt": float
    }
}

Generate parameters strictly following this JSON structure. Each phase should be named 'phase_1' through 'phase_5' exactly as shown."""


class RedisQueueManager:
    def __init__(self, api_key: str, redis_config: dict, num_workers: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.num_workers = num_workers
        self.redis_client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db']
        )
        self.queue_name = redis_config.get('queue', 'maneuver_queue')
        self.is_running = True

    def generate_maneuver(self) -> Optional[ManeuverParameters]:
        """Generate a single maneuver configuration"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": "Generate a new J-Turn maneuver configuration"}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1000
            )

            raw_json = json.loads(response.choices[0].message.content)
            validated = ManeuverParameters(**raw_json)
            return validated

        except Exception as e:
            logging.error(f"Error generating maneuver: {str(e)}")
            return None

    def process_and_enqueue(self, task_id: int) -> bool:
        """Process a single maneuver generation task and enqueue the result"""
        try:
            maneuver = self.generate_maneuver()
            if maneuver:
                # Convert to JSON and push to Redis queue
                maneuver_json = maneuver.model_dump_json()
                self.redis_client.rpush(self.queue_name, maneuver_json)
                logging.info(
                    f"Task {task_id}: Successfully generated and enqueued maneuver")
                return True
            return False
        except Exception as e:
            logging.error(
                f"Task {task_id}: Failed to process or enqueue maneuver: {str(e)}")
            return False

    def generate_maneuvers_batch(self, num_sets: int) -> tuple[int, int]:
        """
        Generate multiple maneuver sets using thread pool
        Returns tuple of (success_count, failure_count)
        """
        success_count = 0
        failure_count = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.process_and_enqueue, i)
                       for i in range(num_sets)]

            for future in tqdm(as_completed(futures),
                               total=num_sets,
                               desc="Generating maneuvers"):
                if future.result():
                    success_count += 1
                else:
                    failure_count += 1

        return success_count, failure_count

    def close(self):
        """Clean up resources"""
        self.redis_client.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate vehicle maneuver parameters using Redis queue")
    parser.add_argument(
        "-n", "--num-sets",
        type=int,
        default=1,
        help="Number of maneuver sets to generate"
    )
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=3,
        help="Number of worker threads"
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis server host"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis server port"
    )
    parser.add_argument(
        "--redis-db",
        type=int,
        default=0,
        help="Redis database number"
    )
    parser.add_argument(
        "--redis-queue",
        default="maneuver_queue",
        help="Redis queue name"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable not set")
        return

    redis_config = {
        'host': args.redis_host,
        'port': args.redis_port,
        'db': args.redis_db,
        'queue': args.redis_queue
    }

    queue_manager = RedisQueueManager(
        api_key=api_key,
        redis_config=redis_config,
        num_workers=args.num_workers
    )

    try:
        logging.info(
            f"Starting maneuver generation with {args.num_sets} sets...")
        success_count, failure_count = queue_manager.generate_maneuvers_batch(
            args.num_sets)

        logging.info(f"Generation complete:")
        logging.info(f"- Successful: {success_count}")
        logging.info(f"- Failed: {failure_count}")
        logging.info(f"- Total processed: {success_count + failure_count}")

    except Exception as e:
        logging.error(f"Critical error during generation: {str(e)}")
    finally:
        queue_manager.close()


if __name__ == "__main__":
    main()
