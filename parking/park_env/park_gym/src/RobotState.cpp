#include "RobotState.h"

static float sqr(float value) { return value * value; }

float RobotState::Distance(RobotDynamics &robot) {
  float sum = 0;
  sum += sqr(x - robot.GetX());
  sum += sqr(y - robot.GetY());
  if (containsTheta) {
    sum += sqr(theta - robot.GetTheta());
  }
  return sum;
}
