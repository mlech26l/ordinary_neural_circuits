#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "neuron.h"
#include <iostream>
#include <vector>
class NeuralNetwork {
  std::vector<Neuron> neurons;

  std::vector<std::pair<int, float>> constNeurons;

public:
  float E_rev_excitatory;
  float E_rev_inhibitory;
  float V_Leak;
  float Sigmoid_mu;
  float Sigmoid_sigma;
  float G_Leak;
  float Default_Cm;

  NeuralNetwork(int size);
  void PrintStats(void);
  void AddExcitatorySynapse(int src, int dest, float weight, float sigma);
  void AddInhibitorySynapse(int src, int dest, float weight, float sigma);
  void AddGapJunction(int src, int dest, float weight);
  void AddConstNeuron(int src, float value);
  std::vector<std::pair<int, float>> &GetConstNeurons() { return constNeurons; }

  void ChangeConstNeuronPotential(int neuronId, float newPotential);
  float GetPotentialOf(int src);
  Synapse *GetSynapse(int src, int dest, SynapseType synType);

  void RemoveSynapse(int src, int dest, SynapseType synType);
  void ForcePotentialOf(int src, float value);

  int Enlarge();
  int GetSize(void);
  int CountSynapses();
  void DoSimulationStep(float deltaT);

  void SetCmOf(int src, float cm);
  void SetVleakOf(int src, float vleak);
  float GetCmOf(int src);
  float GetVleakOf(int src);
  float GetGleakOf(int src);
  void SetGleakOf(int src, float gleak);

  void Reset();
  void PrintSynapsesOf(int dest) {
    std::vector<Synapse> &syns = GetSynapsesOf(dest);
    for (unsigned int i = 0; i < syns.size(); i++) {
      std::cout << "Syn from " << dest << " to " << syns[i].source
                << " of type ";
      if (syns[i].type == Excitatory)
        std::cout << "ex";
      if (syns[i].type == Inhibitory)
        std::cout << "inh";
      if (syns[i].type == GapJunction)
        std::cout << "gj";
      std::cout << std::endl;
    }
  }
  void PrintNeuronDebug(int src);
  float *GetValuePointerOfSynapse(int source, int dest, SynapseType synType);
  float *GetValuePointerOfConstNeuron(int neuron);
  float *GetValuePointerOfCapacity(int neuron);
  float *GetValuePointerOfVleak(int neuron);
  float *GetValuePointerOfGleak(int neuron);
  std::vector<Synapse> &GetSynapsesOf(int dest);

  NeuralNetwork *Clone();

  void CoreDump();
};

#endif
