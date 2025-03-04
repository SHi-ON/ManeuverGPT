from crewai import Task

# Task for Query Enricher Agent: Enrich User Query
enrich_query_task = Task(
    description=(
        "Elaborate and enhance the user's input to optimize the Driver Agent's "
        "task of generating maneuver parameters."
    ),
    expected_output="Enhanced and detailed user query ready for maneuver generation.",
    agent_name="Query Enricher"
)

# Task for Driver Agent: Create Maneuver
create_maneuver_task = Task(
    description=(
        "Generate precise maneuver parameters for a J-Turn maneuver based "
        "on the enriched user input."
    ),
    expected_output="Validated maneuver parameters in JSON format.",
    agent_name="Driver Agent"
)

# Task for Validator Agent: Validate Maneuver
validate_maneuver_task = Task(
    description=(
        "Verify the generated maneuver parameters for correct format, "
        "executability, and high-level quality using Pydantic validation."
    ),
    expected_output="Confirmed valid maneuver parameters or error details.",
    agent_name="Validator Agent"
)
