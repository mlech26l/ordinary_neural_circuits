#ifndef ROBOT_TRACE_H_
#define ROBOT_TRACE_H_

#include "RobotDynamics.h"
#include "RobotState.h"
#include <string>
#include <vector>

class RobotTrace {
private:
  std::vector<RobotState> points;
  int nextPoint;

public:
  RobotTrace();
  void Reset();
  void LoadFromFile(std::string filename);
  void AddPoint(float time, float x, float y, bool containsTheta, float theta);
  bool EndOfTraceReached();
  float GetDifferenceFromTrace(RobotDynamics &robot);
};
#endif
