import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import redis
from common.config import SYSTEM_PROMPT, ManeuverParameters
from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from tqdm import tqdm  # Optional: For progress bars

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Set environment variables
os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o-mini'
os.environ['OPENAI_API_KEY'] = os.getenv(
    'OPENAI_API_KEY', 'Your_OPENAI_API_Key'
)


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Maneuver Orchestrator for CARLA'
    )
    parser.add_argument(
        '--mode',
        choices=['offline', 'online'],
        default='online',
        help='Operation mode: "offline" for testing, "online" for integrating with Redis',
    )
    parser.add_argument(
        '--redis_host',
        default='localhost',
        help='Redis server host (default: localhost)',
    )
    parser.add_argument(
        '--redis_port',
        type=int,
        default=6379,  # Corrected Redis default port
        help='Redis server port (default: 6379)',
    )
    parser.add_argument(
        '--redis_db',
        type=int,
        default=0,
        help='Redis database number (default: 0)',
    )
    parser.add_argument(
        '--redis_queue',
        default='maneuver_queue',
        help='Redis queue name (default: maneuver_queue)',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Number of iterations to generate and enqueue maneuvers (default: 1)',
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=5,
        help='Maximum number of worker threads for parallel processing (default: 5)',
    )
    return parser.parse_args()


def connect_redis(host: str, port: int, db: int) -> redis.Redis:
    """
    Establishes a connection to the Redis server.

    Args:
        host (str): Redis server host.
        port (int): Redis server port.
        db (int): Redis database number.

    Returns:
        redis.Redis: Redis client instance.

    Raises:
        redis.exceptions.ConnectionError: If connection fails.
    """
    try:
        client = redis.Redis(host=host, port=port, db=db)
        client.ping()
        logging.info(f'Connected to Redis at {host}:{port}, DB: {db}')
        return client
    except redis.exceptions.ConnectionError as e:
        logging.error(f'Failed to connect to Redis: {e}')
        raise


def create_agents() -> list:
    """
    Creates and returns a list of Agent instances.

    Returns:
        list: List containing Query Enricher, Driver, and Validator Agents.
    """
    # Creating Agents
    query_enricher_agent = Agent(
        role='Query Enrichment Specialist',
        goal='Extract and clarify all required parameters for maneuver generation',
        backstory=(
            'Expert in analyzing user queries to identify and explicitly request '
            'missing parameters needed for precise maneuver configuration.'
        ),
        verbose=True,
        allow_delegation=False,
    )

    driver_agent = Agent(
        role='Maneuver Generation Engineer',
        goal='Create technically sound maneuver configurations',
        backstory=(
            'Skilled in translating operational requirements into precise '
            'maneuver specifications using domain knowledge.'
        ),
        verbose=True,
        allow_delegation=False,
    )

    validator_agent = Agent(
        role='Maneuver Quality Assurance',
        goal='Ensure safety and executability of maneuvers',
        backstory=(
            'Meticulous validator with expertise in operational constraints '
            'and safety protocols for vehicle maneuvers.'
        ),
        verbose=True,
        allow_delegation=False,
    )

    return [query_enricher_agent, driver_agent, validator_agent]


def create_tasks(iteration_id: int, output_dir: str) -> list:
    """
    Creates and returns a list of Task instances with unique output file paths per iteration.

    Args:
        iteration_id (int): The iteration number.
        output_dir (str): Directory to store output files for this iteration.

    Returns:
        list: List containing Enrich Query, Create Maneuver, and Validate Maneuver Tasks.
    """
    # Creating unique output file paths
    enrich_query_output = os.path.join(
        output_dir, f'enriched_query_{iteration_id}.txt'
    )
    create_maneuver_output = os.path.join(
        output_dir, f'raw_maneuver_{iteration_id}.json'
    )
    validate_maneuver_output = os.path.join(
        output_dir, f'validated_maneuver_{iteration_id}.json'
    )

    # Creating Tasks with Pydantic Integration
    enrich_query_task = Task(
        description=(
            'Analyze and enrich the user request: {user_input}\n'
            'System requirements: {system_prompt}\n'
            'Identify missing parameters and request clarification or '
            'provide reasonable assumptions for missing values.'
        ),
        expected_output=(
            'Structured query containing all parameters required for '
            'maneuver generation in key:value format.'
        ),
        agent=create_agents()[0],
        output_file=enrich_query_output,
    )

    create_maneuver_task = Task(
        description=(
            'Generate J-Turn configuration using these parameters:\n'
            f'{enrich_query_task.output}\n'  # Use the output of enrich_query_task
            'Consider vehicle dynamics and operational constraints.'
        ),
        expected_output='Full maneuver configuration in pure JSON format and no comments or additional text',
        agent=create_agents()[1],
        context=[enrich_query_task],  # Ensure context is set correctly
        output_pydantic_model=ManeuverParameters,
        output_file=create_maneuver_output,
    )

    validate_maneuver_task = Task(
        description=(
            'Validate this maneuver configuration:\n'
            f'{create_maneuver_task.output}\n'  # Use the output of create_maneuver_task
            'Check: 1. Schema compliance 2. Physical feasibility '
            '3. Safety margins 4. Operational constraints'
        ),
        expected_output=(
            'Validated maneuver parameters in JSON format starting like '
            '{phase_1: {throttle: float, steering_angle: float, reverse: bool, brake: float, duration: float} '
            'and so on, with only phase by phase parameters and no comments or additional text.'
        ),
        agent=create_agents()[2],
        context=[create_maneuver_task],  # Ensure context is set correctly
        output_pydantic_model=ManeuverParameters,
        output_file=validate_maneuver_output,
    )

    return [enrich_query_task, create_maneuver_task, validate_maneuver_task]


def create_crew(tasks: list) -> Crew:
    """
    Creates and returns a Crew instance based on provided tasks.

    Args:
        tasks (list): List of Task instances.

    Returns:
        Crew: Configured Crew instance.
    """
    agents = [task.agent for task in tasks]  # Extract agents from tasks

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        process=Process.sequential,
        manager_llm=ChatOpenAI(model='gpt-4o-mini', temperature=0.3),
    )

    return crew


def generate_and_enqueue(args, iteration_id, output_dir):
    """
    Generates and enqueues a single maneuver.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        iteration_id (int): The current iteration number.
        output_dir (str): Directory to store output files for this iteration.
    """
    try:
        tasks = create_tasks(iteration_id, output_dir)
        crew = create_crew(tasks)

        process_inputs = {
            'user_input': 'Generate J-Turn configuration for urban environment',
            'system_prompt': SYSTEM_PROMPT,
        }

        # Start the Crew process
        crew.kickoff(inputs=process_inputs)

        logging.info(
            f'✅ Iteration {iteration_id}: Validated maneuver generated successfully'
        )

        # Read the validated maneuver from file
        validate_maneuver_task = tasks[2]  # The third task is validation
        with open(validate_maneuver_task.output_file, 'r') as f:
            maneuver_json = f.read()

        # Push to Redis if in online mode
        if args.mode == 'online' and args.redis_queue and args.redis_client:
            args.redis_client.rpush(args.redis_queue, maneuver_json)
            logging.info(
                f"✅ Iteration {iteration_id}: Maneuver pushed to Redis queue '{args.redis_queue}'"
            )
        else:
            logging.warning(
                f'Iteration {iteration_id}: Redis client not available. Maneuver not enqueued.'
            )

    except ValidationError as ve:
        logging.error(f'Validation Error in iteration {iteration_id}: {ve}')
    except Exception as e:
        logging.error(
            f'Failed to generate maneuver in iteration {iteration_id}: {e}'
        )


def main():
    args = parse_arguments()

    # Connect to Redis if in online mode
    if args.mode == 'online':
        try:
            redis_client = connect_redis(
                args.redis_host, args.redis_port, args.redis_db
            )
            args.redis_client = redis_client  # Attach to args for later use
        except Exception:
            logging.error('Exiting due to Redis connection failure.')
            return
    else:
        args.redis_client = None

    # Create a unique directory for all iterations' outputs
    output_base_dir = 'maneuver_outputs'
    os.makedirs(output_base_dir, exist_ok=True)

    try:
        logging.info(
            f'Starting {args.mode} mode with {args.iterations} iterations.'
        )

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Prepare a list of futures
            futures = []
            for i in range(1, args.iterations + 1):
                iteration_output_dir = os.path.join(
                    output_base_dir, f'iteration_{i}'
                )
                os.makedirs(iteration_output_dir, exist_ok=True)
                futures.append(
                    executor.submit(
                        generate_and_enqueue, args, i, iteration_output_dir
                    )
                )

            # Optional: Use tqdm to display progress
            for future in tqdm(
                as_completed(futures),
                total=args.iterations,
                desc='Processing Maneuvers',
            ):
                pass  # All logging is handled within generate_and_enqueue

        logging.info(
            f'All {args.iterations} iterations completed successfully.'
        )

    except KeyboardInterrupt:
        logging.info('Orchestrator interrupted by user.')
    finally:
        if args.mode == 'online' and args.redis_client:
            args.redis_client.close()
            logging.info('Redis connection closed.')
        logging.info('Orchestrator terminated.')


if __name__ == '__main__':
    main()
