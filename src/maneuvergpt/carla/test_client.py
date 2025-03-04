import redis
from pydantic import ValidationError

from common.config import ManeuverParameters


def main():
    redis_host = 'localhost'
    redis_port = 6379
    redis_db = 0
    queue_name = 'maneuver_queue'

    try:
        redis_client = redis.Redis(host=redis_host, port=redis_port,
                                   db=redis_db)
        redis_client.ping()
        print(
            f"Connected to Redis at {redis_host}:{redis_port}, DB: {redis_db}")
    except redis.exceptions.ConnectionError as e:
        print(f"Failed to connect to Redis: {e}")
        return

    print(f"Listening to Redis queue '{queue_name}'...")

    while True:
        try:
            _, data = redis_client.blpop(queue_name)
            maneuver_json = data.decode('utf-8')
            try:
                maneuver = ManeuverParameters.parse_raw(maneuver_json)
                print("Received Maneuver Parameters:")
                print(maneuver.model_dump_json(indent=2))
            except ValidationError as ve:
                print(f"Validation Error: {ve}")
        except KeyboardInterrupt:
            print("Test client interrupted by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    print("Disconnected from Redis.")


if __name__ == "__main__":
    main()
