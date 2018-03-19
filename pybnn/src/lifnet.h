#ifndef LIF_NET_H
#define LIF_NET_H

// #define BOOST_LIB_NAME "boost_numpy"
// #include <boost/config/auto_link.hpp>

#include "LifNetConstructor.h"
#include "NeuralInterface.h"
#include "neural_net_serializer.h"
#include "neural_network.h"
#include <boost/python.hpp>
#include <random>
#include <string>
#include <vector>
using namespace boost::python;

class LifNet {
private:
  NeuralNetwork *nn;
  std::vector<NeuralInterfaceInput> inputs;
  std::vector<NeuralInterfaceOutput> outputs;
  std::vector<NeuralBiInterfaceInput> biInputs;
  std::vector<NeuralBiInterfaceOutput> biOutputs;
  std::mt19937 gen;
  std::vector<float *> parameters;
  std::vector<float> backupValues;

  std::vector<float *> sigmaParameters;
  std::vector<float> sigmaBackupValues;

  std::vector<float *> vleakParameters;
  std::vector<float> vleakBackupValues;

  std::vector<float *> gleakParameters;
  std::vector<float> gleakBackupValues;

  std::vector<float *> cmParameters;
  std::vector<float> cmBackupValues;
  float totalTime;

public:
  LifNet(std::string);
  ~LifNet();
  void AddSensoryNeuron(int indx, float input_bound);
  void AddMotorNeuron(int indx, float output_bound);
  void AddBiMotorNeuron(int indxPos, int indxNeg, float min_value,
                        float max_value);
  void AddBiSensoryNeuron(int indxPos, int indxNeg, float min_value,
                          float max_value);
  boost::python::list Update(boost::python::list &inputsArr, float deltaT,
                             int simulationSteps);
  void WriteToFile(std::string);
  void AddNoise(float variance, int samples);
  void AddNoiseSigma(float variance, int samples);
  void AddNoiseVleak(float variance, int samples);
  void AddNoiseGleak(float variance, int samples);
  void AddNoiseCm(float variance, int samples);
  void UndoNoise();
  void CommitNoise();
  void SeedRandomNumberGenerator(int seed);
  void Reset();
  void DumpState(std::string filename);
  void DumpClear(std::string filename);
};

BOOST_PYTHON_MODULE(pybnn) {
  class_<LifNet>("LifNet", init<std::string>())
      .def("AddSensoryNeuron", &LifNet::AddSensoryNeuron)
      .def("AddMotorNeuron", &LifNet::AddMotorNeuron)
      .def("AddBiSensoryNeuron", &LifNet::AddBiSensoryNeuron)
      .def("AddBiMotorNeuron", &LifNet::AddBiMotorNeuron)
      .def("Reset", &LifNet::Reset)
      .def("DumpClear", &LifNet::DumpClear)
      .def("DumpState", &LifNet::DumpState)
      .def("AddNoise", &LifNet::AddNoise)
      .def("AddNoiseSigma", &LifNet::AddNoiseSigma)
      .def("AddNoiseVleak", &LifNet::AddNoiseVleak)
      .def("AddNoiseGleak", &LifNet::AddNoiseGleak)
      .def("AddNoiseCm", &LifNet::AddNoiseCm)
      .def("UndoNoise", &LifNet::UndoNoise)
      .def("CommitNoise", &LifNet::CommitNoise)
      .def("SeedRandomNumberGenerator", &LifNet::SeedRandomNumberGenerator)
      .def("WriteToFile", &LifNet::WriteToFile)
      .def("Update", &LifNet::Update);
  class_<LifNetConstructor>("LifNetConstructor", init<int>())
      .def("AddExcitatorySynapse", &LifNetConstructor::AddExcitatorySynapse)
      .def("AddInhibitorySynapse", &LifNetConstructor::AddInhibitorySynapse)
      .def("AddGapJunction", &LifNetConstructor::AddGapJunction)
      .def("AddConstNeuron", &LifNetConstructor::AddConstNeuron)
      .def("CountSynapses", &LifNetConstructor::CountSynapses)
      .def("WriteToFile", &LifNetConstructor::WriteToFile);
}
#endif /* end of include guard: LIF_NET_H */
