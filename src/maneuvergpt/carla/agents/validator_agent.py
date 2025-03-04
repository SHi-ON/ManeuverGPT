import logging
import sys
from pathlib import Path

from crewai import Agent, Task

sys.path.append(
    str(Path(__file__).parent.parent))  # Add controller directory to path

from crewai import Agent, Task
from pydantic import ValidationError

from common.config import SYSTEM_PROMPT, ManeuverParameters


class ValidatorAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Validator Agent",
            goal="Verify the generated maneuver for format, executability, and quality.",
            backstory=(
                "This agent ensures that all generated maneuvers adhere to the defined "
                "Pydantic models, maintaining high standards of quality and executability."
            ),
            allow_delegation=True,
            verbose=True,
            tools=[]  # Add any necessary tools if needed
        )

    def validate_maneuver(self, raw_data: str) -> ManeuverParameters:
        logging.info("Validator Agent validating maneuver")
        try:
            maneuver = ManeuverParameters.parse_raw(raw_data)
            logging.debug("Validation successful")
            return maneuver
        except ValidationError as e:
            logging.error(f"Validator Agent found invalid maneuver: {e}")
            raise e
