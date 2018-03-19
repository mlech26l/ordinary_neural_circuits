#include "neural_network.h"
#include "neuron.h"

#include <cstddef>
#include <iostream>
NeuralNetwork::NeuralNetwork(int size) : neurons(size) {
  E_rev_excitatory = 0;
  E_rev_inhibitory = -90;
  V_Leak = -70;
  Sigmoid_mu = -40;
  Sigmoid_sigma = 0.1f;
  G_Leak = 1;

  Default_Cm = 0.1;
  Default_Cm = 0.05;

  // Set all neuron potentials to -70mV
  for (size_t i = 0; i < neurons.size(); i++) {
    neurons[i].Initialize(V_Leak, Default_Cm);
    neurons[i].SetGLeak(G_Leak);
  }
}
int NeuralNetwork::Enlarge() {
  int newNeuron = neurons.size();
  Neuron n;
  n.Initialize(V_Leak, Default_Cm);
  neurons.push_back(n);
  return newNeuron;
}
NeuralNetwork *NeuralNetwork::Clone() {
  NeuralNetwork *clone = new NeuralNetwork(GetSize());
  clone->constNeurons = constNeurons;
  for (int i = 0; i < GetSize(); i++) {
    std::vector<Synapse> syns = GetSynapsesOf(i);
    for (size_t j = 0; j < syns.size(); j++) {
      if (syns[j].type == Inhibitory)
        clone->AddInhibitorySynapse(syns[j].source, i, syns[j].w,
                                    syns[j].sigma);
      else if (syns[j].type == Excitatory)
        clone->AddExcitatorySynapse(syns[j].source, i, syns[j].w,
                                    syns[j].sigma);
      else
        clone->AddGapJunction(syns[j].source, i, syns[j].w);
    }
  }
  return clone;
}
void NeuralNetwork::PrintStats(void) {
  std::cout << "Network consits of " << neurons.size() << " neurons and "
            << CountSynapses() << " synapses" << std::endl;
}
void NeuralNetwork::AddExcitatorySynapse(int src, int dest, float weight,
                                         float sigma) {
  neurons[dest].AddSynapse(weight, Excitatory, src, sigma);
}
void NeuralNetwork::AddInhibitorySynapse(int src, int dest, float weight,
                                         float sigma) {
  neurons[dest].AddSynapse(weight, Inhibitory, src, sigma);
}
void NeuralNetwork::AddGapJunction(int src, int dest, float weight) {
  neurons[dest].AddSynapse(weight, GapJunction, src, 1);
  // neurons[src].AddSynapse(weight, GapJunction, dest, 1);
}
void NeuralNetwork::ChangeConstNeuronPotential(int neuronId,
                                               float newPotential) {
  for (unsigned int i = 0; i < constNeurons.size(); i++) {
    if (constNeurons[i].first == neuronId)
      constNeurons[i].second = newPotential;
  }
}
Synapse *NeuralNetwork::GetSynapse(int src, int dest, SynapseType synType) {
  std::vector<Synapse> &synapses = GetSynapsesOf(dest);
  for (size_t i = 0; i < synapses.size(); i++) {
    if (synapses[i].source == src && synapses[i].type == synType)
      return &synapses[i];
  }
  return NULL;
}
float NeuralNetwork::GetPotentialOf(int src) {
  return neurons[src].GetPotential();
}
void NeuralNetwork::DoSimulationStep(float deltaT) {
  for (size_t i = 0; i < neurons.size(); i++) {
    neurons[i].ComputeV_next(deltaT, *this);
  }
  for (size_t i = 0; i < neurons.size(); i++) {
    neurons[i].UseVNext();
  }
  for (size_t i = 0; i < constNeurons.size(); i++) {
    neurons[constNeurons[i].first].ForcePotential(constNeurons[i].second);
  }
}
void NeuralNetwork::RemoveSynapse(int src, int dest, SynapseType synType) {
  neurons[dest].RemoveSynapse(src, synType);
}
int NeuralNetwork::CountSynapses() {
  int count = 0;
  for (size_t i = 0; i < neurons.size(); i++) {
    std::vector<Synapse> &synapses = GetSynapsesOf(i);
    count += synapses.size();
  }
  return count;
}
void NeuralNetwork::AddConstNeuron(int src, float value) {
  std::pair<int, float> p(src, value);
  constNeurons.push_back(p);
}
void NeuralNetwork::ForcePotentialOf(int src, float value) {
  neurons[src].ForcePotential(value);
}
float NeuralNetwork::GetGleakOf(int src) { return neurons[src].GetGleak(); }
void NeuralNetwork::SetGleakOf(int src, float gleak) {
  neurons[src].SetGLeak(gleak);
}
void NeuralNetwork::SetCmOf(int src, float cm) { neurons[src].SetCm(cm); }
void NeuralNetwork::SetVleakOf(int src, float vleak) {
  neurons[src].SetVLeak(vleak);
}
void NeuralNetwork::PrintNeuronDebug(int src) {
  neurons[src].PrintDebug(*this);
}
int NeuralNetwork::GetSize(void) { return neurons.size(); }

void NeuralNetwork::Reset() {
  for (size_t i = 0; i < neurons.size(); i++) {
    neurons[i].ForcePotential(V_Leak);
  }
  for (size_t i = 0; i < constNeurons.size(); i++) {
    neurons[constNeurons[i].first].ForcePotential(constNeurons[i].second);
  }
}
float *NeuralNetwork::GetValuePointerOfSynapse(int source, int dest,
                                               SynapseType synType) {
  return neurons[dest].GetValuePointerOfSynapse(source, synType);
}
float *NeuralNetwork::GetValuePointerOfGleak(int neuron) {
  return neurons[neuron].GetValuePointerOfGleak();
}
float *NeuralNetwork::GetValuePointerOfCapacity(int neuron) {
  return neurons[neuron].GetValuePointerOfCapacity();
}
float *NeuralNetwork::GetValuePointerOfConstNeuron(int neuron) {
  for (size_t i = 0; i < constNeurons.size(); i++) {
    if (neuron == constNeurons[i].first)
      return &constNeurons[i].second;
  }
  return NULL;
}
float NeuralNetwork::GetCmOf(int src) { return neurons[src].GetCm(); }
std::vector<Synapse> &NeuralNetwork::GetSynapsesOf(int dest) {
  return neurons[dest].GetSynapses();
}
float *NeuralNetwork::GetValuePointerOfVleak(int neuron) {
  return neurons[neuron].GetValuePointerOfVleak();
}
float NeuralNetwork::GetVleakOf(int src) { return neurons[src].GetVleak(); }
void NeuralNetwork::CoreDump() {

  for (int i = 0; i < GetSize(); i++) {
    std::vector<Synapse> syns = GetSynapsesOf(i);
    for (size_t j = 0; j < syns.size(); j++) {
      if (syns[j].type == Inhibitory)
        std::cout << "inh ";
      else if (syns[j].type == Excitatory)
        std::cout << "ex ";
      else
        std::cout << "gj ";
      std::cout << syns[j].source << " " << i << " " << syns[j].w << std::endl;
    }
  }
}
