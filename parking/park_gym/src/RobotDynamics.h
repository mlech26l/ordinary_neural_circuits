#ifndef ROBOT_DYNAMICS_H_
#define ROBOT_DYNAMICS_H_

#include <cstddef>
#include <queue>
#include <utility>

class RobotDynamics {
private:
  float Acceleration = 0.31;
  float AngularAcceleration = 1.25;

  float totalTime;
  float x;
  float y;
  float theta;
  float v_want;
  float v_is;

  // 150 ms
  float linearActuatorDelay = 0.150f;
  // 100 ms
  float angularActuatorDelay = 0.100f;

  // Pair stores <Valid Time, Value> to implement delay of arbitrary length
  std::queue<std::pair<float, float>> angularActuatorCommandQueue;
  std::queue<std::pair<float, float>> linearActuatorCommandQueue;
  float w_want;
  float w_is;

  void CheckActuatorCommandQueue();

public:
  RobotDynamics();
  void Reset();
  void DoSimulationStep(float deltaT);
  void SendCommands(float linearVelocity, float angularVelocity);
  float GetVIs() { return v_is; }
  float GetWIs() { return w_is; }
  float GetX() { return x; }
  float GetY() { return y; }
  float GetTotalTime() { return totalTime; }
  float GetTheta() { return theta; }
  void SetAcceleration(float newAcceleration) {
    Acceleration = newAcceleration;
  }
  void SetAngularAcceleration(float newAngularAcceleration) {
    AngularAcceleration = newAngularAcceleration;
  }
};
#endif
