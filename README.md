# Neurovis3D a ROS compatible neural network visualizer

Neurovis3D is a python neural network visualizer, that runs on Panda 3D. It opens up a ROS node called "NeuroVis3D", to which you can publish topics, the only mandatory topic is "/neurovis/neuronName". The simulation runs in real-time, and I included a few examples. The colors of neurons represent neuron activity (neuron outputs). It is built as a 3D upgrade for Zumos original neurovis. https://gitlab.com/zumoarthicha/neurovis

Most of the relevant code can be found in https://github.com/tevzek/Nvis3DStandalone/tree/master/catkin_ws/src/neurovis3dstanalone/scripts/Nvis3D

## ROS interface

The project is build with a ROS Interface, enabeling easy integration with ROS related projects. The neurovis node subscribes to the following topics. 

![Nvis graph](https://github.com/tevzek/Nvis3DStandalone/blob/master/ReadMePics/graph.png?raw=true)

A more descriptive table is seen down below

| Topic name                  | Type              | Purpuse                                                   | Exsample                            | Note                                                                    |
|-----------------------------|-------------------|-----------------------------------------------------------|-------------------------------------|-------------------------------------------------------------------------|
| /neurovis/neuronName        | String            | Initilisez the neurons                                    | "A/B/C/"                            | "Name1/Name2/ ..."                                                      |
| /neurovis/neuronPos         | String            | Sets the positions                                        | "A_0_0_0/B_1_0_0/C_3_2.5_1/"        | "LayerName1_Layer(Z)_X_Y/ ...."                                         |
| /neurovis/connections       | Float32MultiArray | Sets the connections                                      | [[0,1,1],[1,0,1],[1,1,0]]           | Connection matrix neuron N in colum N is connected to neuron M in row M |
| /neurovis/connectionsLayers | String            | Sets the connections, but only for fully connected layers | "A-B/B-C/C-A/"                      | "LayerName1-LayerName2/ ..."                                            |
| /neurovis/activity          | Float32MultiArray | Updates the activity                                      | [0,0,1]                             | [act1,act2,act3]                                                        |
| /neurovis/createDisplay     | Int32MultiArray   | Creates a graphical display with id and dimentisons       | [0,2,2]                             | [id,X,Y]                                                                |
| /neurovis/updateDisplay     | Int32MultiArray   | Updates the display with id                               | [0,0,0,255,0,255,0,0,0,100,100,100] | [id, r1, g1, b1, r2, b2, g2 ...]                                        |
| /neurovis/setDisplayText    | String            | Updates the text on the display                           | "Hello world"                       | "String to display"                                                     |

##Setup

the setup is very similar to the original neurovis, the project was made on Ubuntu 18 with ROS melodic.

1. Robot Operating System (ROS) under Ubuntu Operating System (test with Ubuntu 18 with ROS melodic)
2. Python 3 with these modules TODO make req.txt
	- pyglet
	- pyopengl
	- numpy
	- opencv
	- bezier
	- networkx
	- rospkg

Note that these modules can be installed via the command:
```bash
pip install <module(s)>
```
Alternatively, you can download the given requirement file (requirements.txt) and use:
```bash
pip install -r requirements.txt
```

3. Download and extract Nvis3DStandalone folder 
4. To test whether the installation is successful or not, first start a roscore in one terminal:
```bash
roscore
```
5. Navigate to your project to the "Nvis3DStandalone" folder. Inside the folder, you will find "src" subfolder (and "build" and "devel" subfolders, if you have built it before). If you have yet built the program, use:
In another one:
```bash
catkin_make
```
6. Next, use another command at add the program path to system path (you have do this every time you open the terminal). Alternatively, to make your computer run this command automatically after open new terminal, you can add this command to .bashrc
```bash
source <path_to_your_project>/neurovis/devel/setup.bash
```
or
```bash
nano ~/.bashrc
```
and open new terminal

7. Run the neurovis node with python 3
```bash
python Nvis3DStandalone/catkin_ws/src/neurovis3dstanalone/scripts/Nvis3D/NvisNode.py
```

8. There are four examples in "Nvis3DStandalone/catkin_ws/src/neurovis3dstanalone/scripts/Examples/". To run the examples use:
```bash
python pytorch1.py
```
##Functionalities

Below You can see an image explainging the functionalities

![Nvis graph](https://github.com/tevzek/Nvis3DStandalone/blob/master/ReadMePics/NumberedImage.jpg?raw=true)

1 - You can click and drag neurons around. The selected neuron changes color to blue. The connections are bezier curves. Neurons change their color from red to green depending on the activity level in the range of [-1,1]

2 - Once the neuron is selected you can see its data on the left. It graphicaly displayis inputs and outputs of the neuron, the activity, in/out connections and its name.

3 - On the right is the tab controller. The clicked tab will open up and give you its functionalities. In the picture the layers tab is open.

4 - Corelation tab. It allowes the user to see the pearson correlation between the timeseries of neurons. The user can set the step, corelation cutoff and the time series size. The corelated neurons will drift twoards each other. By default the correlation is mesured only for the neurons connected to the mesured neuron, but this can be changed to mesure globaly.

5 - Input tab. Shows the activity of inputs, neurons with names starting with I. Made kinda obsolete by the Displays tab.

6 - Pushing tab. Makes neurons push each other away with a certian amount of strenght set by the user. So they don't group togeather. The radius of the soft and hard wall can be set. Soft wall pushes them away with a certian strenght set by the user and hard wall does not allow them to touch in this radius. You can also toggle the "show wall" functionality that visulizes the walls on a selected neuron. Note, using this functionalitis the neurons might want to drift into infinity. 

7 - Micelanius tab. Micelanius functionalities, like randomly scatering neurons and hiding connections for better performance. 

8 - Display tab. A tab that shows images send by ROS. They stack up verticaly.



