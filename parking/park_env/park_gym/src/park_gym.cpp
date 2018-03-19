#include "park_gym.h"

#include <boost/python.hpp>

float ParkGym::Clamp(float x) {
  if (x > 1.0f)
    return 1.0f;
  else if (x < -1.0f)
    return -1.0f;
  return x;
}
ParkGym::ParkGym() {}
void ParkGym::LoadReferenceTrace(std::string filename) {
  referenceTrace.LoadFromFile(filename);
}
void ParkGym::Reset() {
  robot.Reset();
  referenceTrace.Reset();
}
void ParkGym::UpdatePhysics(int numberOfSteps) {
  for (int i = 0; i < numberOfSteps; i++) {
    robot.DoSimulationStep(deltaT);
  }
}
float ParkGym::GetReward() {
  return -referenceTrace.GetDifferenceFromTrace(robot);
}
bool ParkGym::IsDone() { return referenceTrace.EndOfTraceReached(); }
void ParkGym::Actuate(float linearVelocity, float angularVelocity) {
  linearVelocity = Clamp(linearVelocity);
  angularVelocity = Clamp(angularVelocity);

  angularVelocity *= SteeringSpeed;
  if (linearVelocity >= 0.0f)
    linearVelocity *= MotorSpeedForward;
  else
    linearVelocity *= MotorSpeedBackward;

  robot.SendCommands(linearVelocity, angularVelocity);
}
float ParkGym::GetX() { return robot.GetX(); }
float ParkGym::GetY() { return robot.GetY(); }
float ParkGym::GetTheta() { return robot.GetTheta(); }
