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
- First you need to install the Reinforcement Learning environments: [Open-AI Gym](https://gym.openai.com/) [Roboschool](https://github.com/openai/roboschool) and [rllab](https://github.com/rll/rllab)
- Next you need to compile the **pybnn** library (simulates neuronal circuits using C++) and copy the shared object to each of the local working directories. You have to install python-boost:
```bash
sudo apt-get install libboost-python-dev
```
and then compile and copy the library:
```bash
cd pybnn
make
cp bin/pybnn.so ../cartpole_rllab/
cp bin/pybnn.so ../mountaincar_rllab/
cp bin/pybnn.so ../invpend_roboschool/
cp bin/pybnn.so ../mountaincar_gym/
cp bin/pybnn.so ../parking/
```

## Verifying
To check if the toolchain is working as intended you can execute a learned neuronal policy:
```bash
cd mountaincar_gym/
python3 mountaincartw.py
```
or
```bash
cd invpend_roboschool/
python3 twcenterpend.py
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
