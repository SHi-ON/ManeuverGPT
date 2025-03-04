from pydantic import BaseModel, Field
from typing import List, Dict, Optional

SYSTEM_PROMPT = """
You are an autonomous vehicle maneuver configuration generator tasked with creating precise parameters for a J-Turn maneuver. The maneuver consists of 5 distinct phases, each with specific behaviors to ensure a controlled and safe execution.

### Parameter Ranges and Increments:
- **Throttle**: 0.0 to 1.0 (increments of 0.1)
- **Steering Angle**: -1.0 to 1.0 (increments of 0.2)
- **Brake**: 0.0 to 1.0 (increments of 0.1)
- **Duration**: 0.0 to 5.0 seconds (increments of 0.3 seconds)

### Phases and Requirements:
from typing import List, Dict
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

Generate parameters strictly following this JSON structure. Each phase should be named 'phase_1' through 'phase_5' exactly as shown.
"""

class PhaseParameters(BaseModel):
    throttle: float = Field(..., ge=0.0, le=1.0, description="Throttle value between 0.0 and 1.0")
    steering_angle: float = Field(..., ge=-1.0, le=1.0, description="Steering angle between -1.0 and 1.0")
    reverse: bool = False
    brake: float = Field(..., ge=0.0, le=1.0, description="Brake value between 0.0 and 1.0")
    duration: float = Field(..., ge=0.0, le=5.0, description="Duration in seconds")

class ManeuverParameters(BaseModel):
    phase_1: PhaseParameters
    phase_2: PhaseParameters
    phase_3: PhaseParameters
    phase_4: PhaseParameters
    phase_5: PhaseParameters
    success_conditions: Optional[Dict[str, float]] = None  # Make optional

    class Config:
        json_schema_extra = {
            "example": {
                "phase_1": {
                    "throttle": 0.9,
                    "steering_angle": 0.0,
                    "reverse": True,
                    "brake": 0.0,
                    "duration": 3.0
                },
                "phase_2": {
                    "throttle": 0.7,
                    "steering_angle": 0.9,
                    "reverse": True,
                    "brake": 0.0,
                    "duration": 0.6
                },
                "phase_3": {
                    "throttle": 0.9,
                    "steering_angle": -0.5,
                    "reverse": False,
                    "brake": 0.0,
                    "duration": 0.4
                },
                "phase_4": {
                    "throttle": 0.5,
                    "steering_angle": 0.0,
                    "reverse": False,
                    "brake": 0.0,
                    "duration": 1.2
                },
                "phase_5": {
                    "throttle": 0.0,
                    "steering_angle": 0.0,
                    "reverse": False,
                    "brake": 0.9,
                    "duration": 0.0
                },
               
            }
        }

class ManeuverCollection(BaseModel):
    maneuvers: List[ManeuverParameters]
