#include "neuron.h"
#include "neural_network.h"
#include "neuron_util.h"
#include <iostream>
#include <stdlib.h>
void Neuron::Initialize(float v_eq, float cm) {
  v = v_eq;
  Vleak = v_eq;
  Cm = cm;
}
void Neuron::AddSynapse(float w, SynapseType type, int src, float sigma) {
  for (size_t i = 0; i < synapses.size(); i++) {
    if (synapses[i].source == src && synapses[i].type == type) {
      std::cerr << "Assertion! Multiple synapse of same type: " << src
                << std::endl;
      return;
    }
  }
  synapses.push_back(Synapse(w, type, src, sigma));
}
// float Neuron::GetInflowCurrent(NeuralNetwork &nn) {
//   float current = 0;
//   float excitationFactor = 0;
//   float inhibitionFactor = 0;
//   for (size_t i = 0; i < synapses.size(); i++) {
//     if (synapses[i].type == Excitatory) {
//       excitationFactor +=
//           synapses[i].w * sigmoid(nn.GetPotentialOf(synapses[i].source),
//                                   nn.Sigmoid_mu, nn.Sigmoid_sigma);
//     } else if (synapses[i].type == Inhibitory) {
//       inhibitionFactor +=
//           synapses[i].w * sigmoid(nn.GetPotentialOf(synapses[i].source),
//                                   nn.Sigmoid_mu, nn.Sigmoid_sigma);
//     } else if (synapses[i].type == GapJunction) {
//       current += synapses[i].w * (nn.GetPotentialOf(synapses[i].source) - v);
//     }
//   }
//   current += excitationFactor * (nn.E_rev_excitatory - v);
//   current += inhibitionFactor * (nn.E_rev_inhibitory - v);
//   return current;
// }
void Neuron::ComputeV_next(float deltaT, NeuralNetwork &nn) {
  float lhs = Cm / deltaT + Gleak;
  float rhs = v * Cm / deltaT + Gleak * Vleak;

  for (size_t i = 0; i < synapses.size(); i++) {
    if (synapses[i].type == Excitatory) {
      float w_ex =
          synapses[i].w * sigmoid(nn.GetPotentialOf(synapses[i].source),
                                  nn.Sigmoid_mu, synapses[i].sigma);
      lhs += w_ex;
      rhs += w_ex * nn.E_rev_excitatory;
    } else if (synapses[i].type == Inhibitory) {
      float w_inh =
          synapses[i].w * sigmoid(nn.GetPotentialOf(synapses[i].source),
                                  nn.Sigmoid_mu, synapses[i].sigma);
      lhs += w_inh;
      rhs += w_inh * nn.E_rev_inhibitory;
    } else if (synapses[i].type == GapJunction) {
      float w_gj = synapses[i].w;
      lhs += w_gj;
      rhs += w_gj * nn.GetPotentialOf(synapses[i].source);
    }
  }

  v_next = rhs / lhs;
}
void Neuron::RemoveSynapse(int source, SynapseType synType) {
  for (size_t i = 0; i < synapses.size(); i++) {
    if (synapses[i].source == source &&
        synapses[i].type == synType) { // erase the i-th element
      synapses.erase(synapses.begin() + i);
      return;
    }
  }
}
void Neuron::UseVNext(void) { v = v_next; }
void Neuron::ForcePotential(float value) {
  v = value;
  v_next = value;
}
float Neuron::GetPotential(void) { return v; }
void Neuron::PrintDebug(NeuralNetwork &nn) {
  // float current = 0;
  // float excitationFactor = 0;
  // float inhibitionFactor = 0;
  // for (size_t i = 0; i < synapses.size(); i++) {
  //   if (synapses[i].type == Excitatory) {
  //     float fact =
  //         synapses[i].w * sigmoid(nn.GetPotentialOf(synapses[i].source),
  //                                 nn.Sigmoid_mu, nn.Sigmoid_sigma);
  //     std::cout << "Excitatory Synapse from " << synapses[i].source
  //               << " factor: " << fact << std::endl;
  //     excitationFactor += fact;
  //
  //   } else if (synapses[i].type == Inhibitory) {
  //     float fact =
  //         synapses[i].w * sigmoid(nn.GetPotentialOf(synapses[i].source),
  //                                 nn.Sigmoid_mu, nn.Sigmoid_sigma);
  //     std::cout << "Inhibitory Synapse from " << synapses[i].source
  //               << " factor: " << fact << std::endl;
  //     inhibitionFactor += fact;
  //   } else if (synapses[i].type == GapJunction) {
  //     std::cout << "Gap junction from " << synapses[i].source << " current: "
  //               << synapses[i].w * (nn.GetPotentialOf(synapses[i].source) -
  //               v)
  //               << std::endl;
  //
  //     current += synapses[i].w * (nn.GetPotentialOf(synapses[i].source) - v);
  //   }
  // }
  // current += excitationFactor * (nn.E_rev_excitatory - v);
  // current += inhibitionFactor * (nn.E_rev_inhibitory - v);
  //
  // float v_next_fake =
  //     1.0f / (nn.G_Leak + Cm / 0.1) *
  //     (Cm / 0.1 * v + nn.G_Leak * nn.V_Leak + GetInflowCurrent(nn));
  // std::cout << "Fake vnext: " << v_next_fake << std::endl;
}

float *Neuron::GetValuePointerOfSynapse(int source, SynapseType synType) {
  for (size_t i = 0; i < synapses.size(); i++) {
    if (synapses[i].source == source && synapses[i].type == synType)
      return &synapses[i].w;
  }
  return NULL;
}

float *Neuron::GetValuePointerOfSigma(int source, SynapseType synType) {
  for (size_t i = 0; i < synapses.size(); i++) {
    if (synapses[i].source == source && synapses[i].type == synType)
      return &synapses[i].sigma;
  }
  return NULL;
}
