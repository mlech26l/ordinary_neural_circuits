#ifndef NEURON_H_
#define NEURON_H_

#include "synapse.h"
#include <vector>

class NeuralNetwork;
class Neuron {
private:
  float v;
  float v_next;
  float Cm;
  float Vleak;
  float Gleak;
  std::vector<Synapse> synapses;

public:
  void Initialize(float v_eq, float cm);
  float GetCm() { return Cm; }
  void SetCm(float val) { Cm = val; }
  void SetVLeak(float val) { Vleak = val; }
  float GetVleak() { return Vleak; }
  void SetGLeak(float val) { Gleak = val; }
  float GetGleak() { return Gleak; }

  void AddSynapse(float w, SynapseType type, int src, float sigma);
  std::vector<Synapse> &GetSynapses() { return synapses; }
  // float GetInflowCurrent(NeuralNetwork &nn);
  void ComputeV_next(float deltaT, NeuralNetwork &nn);

  void UseVNext(void);
  void ForcePotential(float value);
  float GetPotential(void);
  void PrintDebug(NeuralNetwork &nn);

  void RemoveSynapse(int source, SynapseType synType);
  float *GetValuePointerOfSynapse(int source, SynapseType synType);
  float *GetValuePointerOfCapacity() { return &Cm; }

  float *GetValuePointerOfSigma(int source, SynapseType synType);
  float *GetValuePointerOfVleak() { return &Vleak; }
  float *GetValuePointerOfGleak() { return &Gleak; }
};

#endif
