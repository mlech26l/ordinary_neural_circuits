import numpy as np
import os


if(not os.path.exists("circuits")):
    os.makedirs("circuits")


def gen_circuit(seed):
    
    rnd = np.random.RandomState(2373*seed+7*seed*seed+17*21)
    with open(os.path.join("circuits","circuit_{:03d}.bnn".format(seed)), "w") as f:
        f.write("size 11\n")

        for i in range(11):
            f.write("cm {:d} 0.05 11\n".format(i))
            
        valid_dest = [2,3,4,5,8,9,10]
        valid_types = ["ex","inh","gj"]
        for i in range(28):
            src = rnd.randint(0,11)
            dest_index = rnd.randint(0,len(valid_dest))
            dest = valid_dest[dest_index]

            # Synapse type
            syn_index = rnd.randint(0,len(valid_types))
            syn = valid_types[syn_index]

            w_init = rnd.uniform(0,2)

            f.write("{} {:d} {:d} {:0.4f}\n".format(syn,src,dest,w_init))



for i in range(30):
    gen_circuit(i)