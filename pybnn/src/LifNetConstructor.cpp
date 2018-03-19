#include "LifNetConstructor.h"

LifNetConstructor::LifNetConstructor(int size) { nn = new NeuralNetwork(size); }
void LifNetConstructor::AddExcitatorySynapse(int source, int dest, float weight,
                                             float sigma) {
  nn->AddExcitatorySynapse(source, dest, weight, sigma);
}
void LifNetConstructor::AddInhibitorySynapse(int source, int dest, float weight,
                                             float sigma) {
  nn->AddInhibitorySynapse(source, dest, weight, sigma);
}
void LifNetConstructor::AddGapJunction(int source, int dest, float weight) {
  nn->AddGapJunction(source, dest, weight);
}
void LifNetConstructor::AddConstNeuron(int nid, float potential) {
  nn->AddConstNeuron(nid, potential);
}

void LifNetConstructor::WriteToFile(std::string filename) {
  NeuralNetSerializer serializer(filename);
  serializer.Serialize(nn);
}
int LifNetConstructor::CountSynapses() { return nn->CountSynapses(); }
