import logging
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent.parent))  # Add controller directory to path

from crewai import Agent, Task
from pydantic import ValidationError

from common.config import SYSTEM_PROMPT, ManeuverParameters
import json


class DriverAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Driver Agent",
            goal="Create maneuver parameters from enriched user input.",
            backstory=(
                "This agent is responsible for generating precise maneuver parameters based "
                "on the enhanced queries provided by the Query Enricher Agent."
            ),
            allow_delegation=True,
            verbose=True,
            tools=[]  # Add any necessary tools if needed
        )

    def create_maneuver(self, enriched_query: str) -> ManeuverParameters:
        logging.info("Driver Agent creating maneuver")
        try:
            response = self.client.generate(enriched_query,
                                            prompt=SYSTEM_PROMPT)
            raw_json = json.loads(response)
            logging.debug(f"Generated Maneuver JSON: {raw_json}")
            maneuver = ManeuverParameters(**raw_json)
            logging.info("Maneuver creation successful")
            return maneuver
        except (ValidationError, json.JSONDecodeError) as e:
            logging.error(f"Driver Agent failed to create maneuver: {e}")
            raise e
