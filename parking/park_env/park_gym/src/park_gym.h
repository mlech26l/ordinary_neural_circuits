#ifndef PARK_GYM_H_
#define PARK_GYM_H_

#include "RobotDynamics.h"
#include "RobotTrace.h"
#include <cmath>
#include <string>

#include <boost/python.hpp>
using namespace boost::python;
class ParkGym {
private:
  RobotDynamics robot;
  RobotTrace referenceTrace;

  float deltaT = 0.01f;
  float SteeringSpeed = 0.18;

  float AreaSize = 2;
  float MotorSpeedForward = 0.16;
  float MotorSpeedBackward = 0.16;
  float ThetaMax = M_PI / 2;

  float Clamp(float x);

public:
  ParkGym();
  void Reset();
  void LoadReferenceTrace(std::string filename);
  void UpdatePhysics(int numberOfSteps);
  void Actuate(float linearVelocity, float angularVelocity);
  bool IsDone();
  float GetReward();
  float GetX();
  float GetY();
  float GetTheta();
};

BOOST_PYTHON_MODULE(pyparkgym) {
  class_<ParkGym>("ParkGym")
      .def("Reset", &ParkGym::Reset)
      .def("LoadReferenceTrace", &ParkGym::LoadReferenceTrace)
      .def("UpdatePhysics", &ParkGym::UpdatePhysics)
      .def("IsDone", &ParkGym::IsDone)
      .def("Actuate", &ParkGym::Actuate)
      .def("GetReward", &ParkGym::GetReward)
      .def("GetX", &ParkGym::GetX)
      .def("GetY", &ParkGym::GetY)
      .def("GetTheta", &ParkGym::GetTheta);
}

#endif /* end of include guard: PARK_GYM_H_ */
