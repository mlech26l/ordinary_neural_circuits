#ifndef NEURAL_NET_SERIALIZER_H_
#define NEURAL_NET_SERIALIZER_H_

#include "neural_network.h"
#include <string>
class NeuralNetSerializer {
private:
  std::string filename;

public:
  NeuralNetSerializer(std::string file);
  NeuralNetwork *Deserialize();
  void Serialize(NeuralNetwork *nn);
};
#endif
