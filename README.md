# Neurovis3D a ROS compatible neural network visualizer

Neurovis3D is a python neural network visualizer, that runs on Panda 3D. It opens up a ROS node called "NeuroVis3D", to which you can publish topics, the only mandatory topic is "/neurovis/neuronName". The simulation runs in real-time, and I included a few examples. The colors of neurons represent neuron activity (neuron outputs). It is built as a 3D upgrade for Zumos original neurovis. https://gitlab.com/zumoarthicha/neurovis

Most of the relevant code can be found in https://github.com/tevzek/Nvis3DStandalone/tree/master/catkin_ws/src/neurovis3dstanalone/scripts/Nvis3D

