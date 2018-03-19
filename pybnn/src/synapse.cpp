#include "synapse.h"

Synapse::Synapse(float weight, SynapseType typ, int src, float _sigma) {
  w = weight;
  type = typ;
  source = src;
  sigma = _sigma;
}
