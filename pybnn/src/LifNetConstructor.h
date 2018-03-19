#ifndef LIFNET_CONSTRUCTOR_
#define LIFNET_CONSTRUCTOR_

#include "neural_net_serializer.h"
#include "neural_network.h"
#include <boost/python.hpp>
#include <string>

class LifNetConstructor {
private:
  NeuralNetwork *nn;

public:
  LifNetConstructor(int size);
  ~LifNetConstructor() { delete nn; }
  void AddExcitatorySynapse(int source, int dest, float weight, float sigma);
  void AddInhibitorySynapse(int source, int dest, float weight, float sigma);
  void AddGapJunction(int source, int dest, float weight);
  void AddConstNeuron(int nid, float potential);
  void WriteToFile(std::string filename);
  int CountSynapses();
};
#endif /* end of include guard: LIFNET_CONSTRUCTOR_ */
