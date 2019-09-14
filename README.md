# Neuronal Circuit Policies
Official code repository to verify and reproduce the experiments of the paper *Re-purposing Compact Neuronal Circuit Policies to Govern Reinforcement Learning Tasks*
```bibtex
@article{hasani2018repurposing,
  title={Re-purposing Compact Neuronal Circuit Policies to Govern Reinforcement Learning Tasks},
  author={Hasani, Ramin M. and Lechner, Mathias  and  Amini, Alexander and Rus,Daniela and Grosu, Radu},
  journal={arXiv preprint arXiv:1809.04423},
  year={2018}
}
```

Detailed Description of the experiments can be found in the supplementary materials section of the paper, here: https://arxiv.org/pdf/1809.04423.pdf

# How to get it working:
- First you need to install the Reinforcement Learning environments: [Open-AI Gym](https://gym.openai.com/) [Roboschool](https://github.com/openai/roboschool) and [mujoco-py](https://github.com/openai/mujoco-py)
- Next you need to compile the **pybnn** library (simulates neuronal circuits using C++) and copy the shared object to each of the local working directories. You have to install python-boost:
```bash
sudo apt-get install libboost-python-dev
```
and then compile and copy the library:
```bash
cd pybnn
make
cp bin/pybnn.so ../invpend_roboschool/
cp bin/pybnn.so ../mountaincar_gym/
cp bin/pybnn.so ../half_cheetah/
cp bin/pybnn.so ../parking/
```

## Verifying
To check if the toolchain is working as intended you can execute a learned neuronal policy:
```bash
cd twmountaincar/
python3 twmountaincar.py --file tw-optimized.bnn
```
or
```bash
cd invpend_roboschool/
python3 twcenterpend.py --file tw-optimized.bnn
```
or
```bash
cd half_cheetah/
python3 half_cheetah.py --file tw-optimized.bnn
```

*Note:* You need to `cd` into the particular directories because the library and the optimized policy parameters are loaded from relative paths of the working directory.

## Parking
To get the deterministic parking environment working you have to compile the rover simulator:
```bash
cd parking/park_gym/
make
cp bin/pyparkgym.so ../
cd ..
```

## Random circuits
To generate random circuits that have the same size as the TW circuit run
```bash
cd generate_circuits/
python3 generate_circuit.py
```

## Other NN architectures (MLP, LSTM)
To train a MLP or LSTM on the tasks by our Adaptive Random Search run
```bash
cd other_nn_architectures/
python3 nn_run_env.py --env [invpend|cheetah|mountaincar]
```