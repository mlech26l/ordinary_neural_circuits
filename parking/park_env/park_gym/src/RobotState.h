#ifndef ROBOT_STATE_H_
#define ROBOT_STATE_H_

#include "RobotDynamics.h"

class RobotState {
private:
  float t;
  float x;
  float y;

  bool containsTheta;
  float theta;

public:
  RobotState(float _time, float _x, float _y, bool _containsTheta,
             float _theta) {
    x = _x;
    y = _y;
    t = _time;
    containsTheta = _containsTheta;
    theta = _theta;
  }
  float Distance(RobotDynamics &robot);
  float GetTimePoint(void) { return t; }
  float GetX() { return x; }
  float GetY() { return y; }
  float GetTheta() { return theta; }
  bool GetContainsTheta() { return containsTheta; }
};
#endif
