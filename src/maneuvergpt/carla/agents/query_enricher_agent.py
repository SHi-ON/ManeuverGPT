from crewai import Agent


class QueryEnricherAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Query Enricher',
            goal="Elaborate and enhance user input to optimize the Driver Agent's work.",
            backstory=(
                'This agent specializes in refining and expanding user queries, '
                'ensuring that the Driver Agent receives detailed and optimized input for maneuver generation.'
            ),
            allow_delegation=True,
            verbose=True,
            tools=[],  # Add any necessary tools if needed
        )
