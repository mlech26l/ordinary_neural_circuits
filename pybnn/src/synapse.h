#ifndef SYNAPSE_H_
#define SYNAPSE_H_

enum SynapseType { Excitatory, Inhibitory, GapJunction };
class Synapse {
public:
  float w;
  SynapseType type;
  int source;
  float sigma;
  Synapse(float weight, SynapseType typ, int src, float _sigma);
};

#endif
