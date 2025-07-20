import math


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[: truncate - 1] + '\u2026') if len(name) > truncate else name


def world_to_body(v, yaw_deg):
    """Rotate CARLA world-frame velocity into the vehicle body frame."""
    yaw_rad = math.radians(yaw_deg)
    v_long = v.x * math.cos(yaw_rad) + v.y * math.sin(yaw_rad)
    v_lat = -v.x * math.sin(yaw_rad) + v.y * math.cos(yaw_rad)
    return v_long, v_lat
