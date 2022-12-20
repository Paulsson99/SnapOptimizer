# SNAP Optimizer
A package that can be used to optimze sequences of SNAP-Displacement gates for quantum control over a 3D cavity

# Requirements
Some packages required by the optimizer only exists for Unix environments. So you can´t run this code on a Windows computer. 

To optimze pulses the library **qctrl** is required. You will need an account for this to work. If you don´t need to optimze the pulses you can ignore this. 

# Installation
Begin by cloning this project and creating a new virtual environment in your prefered way. Activate the environment and run

    # To only install the SNAP-Displacement optimizer
    pip install -e .

    # To also install the pulse optimizer, instead run
    pip install -e ".[pulses]"

# Usage

