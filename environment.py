from enum import Enum


class Environment(str, Enum):
    SafetyPointGoalEasy = "OfflinePointGoal1-v0"
    SafetyPointButtonEasy = "OfflinePointButton1-v0"
    SafetyPointPushEasy = "OfflinePointPush1-v0"
    SafetyPointCircleEasy = "OfflinePointCircle1-v0"
    SafetyCarGoalEasy = "OfflineCarGoal1-v0"
    SafetyCarButtonEasy = "OfflineCarButton1-v0"
    SafetyCarPushEasy = "OfflineCarPush1-v0"
    SafetyCarCircleEasy = "OfflineCarCircle1-v0"
    SafetyHopperVelocity = "OfflineHopperVelocity-v1"
    SafetyWalkerVelocity = "OfflineWalker2dVelocity-v1"


ENVIRONMENT_MAX_TIMESTEP = {
    Environment.SafetyPointGoalEasy: 1000,
    Environment.SafetyPointButtonEasy: 1000,
    Environment.SafetyPointPushEasy: 1000,
    Environment.SafetyPointCircleEasy: 500,
    Environment.SafetyCarGoalEasy: 1000,
    Environment.SafetyCarButtonEasy: 1000,
    Environment.SafetyCarPushEasy: 1000,
    Environment.SafetyCarCircleEasy: 500,
    Environment.SafetyHopperVelocity: 1000,
    Environment.SafetyWalkerVelocity: 1000,
}
