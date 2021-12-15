# 28_PBD_Hair

This is a group project for physics-based-simulation course at ETH Zurich. In the group project, a CG simulation using taichi framework is recommended. For the ease of rendering, we turn to the coding block inside Blender. Thus, this simulation can only be played with the file 28_pbd_hair_code.blend. The file 28_pbd_hair_code.py is only for display. 

The chosen topic is hair simulation with specific methods of:
1. Position-based-dynamics(PBD)
2. Dynamic-Follow-The-Leader(DFTL)
3. Volumetric representation for hair-hair interaction
4. Signed distance field for collision
5. Transformation for shape constraints(global + local)


## How to install

1. Install Blender 2.93.5

2. Intall Taichi Blender Integration. Link can be found below:

https://github.com/taichi-dev/taichi_blend


## How to play 

1. Open the file 28_pbd_hair_code.blend in Blender. 

2. Go to the ``Scripting`` window, press the play button to run the script. Blender may stuck a while for the first launch.

3. Go back to ``Layout`` window. Press SPACE and you should see hair moving. The animation may suffer from low FPS on the first time running. 

## Result

1. Short hair with global stiffness = 0.001:  
![grab-landing-page](https://github.com/loOong-Cheng/HairSimulation--Physics-based-simulation-of-computer-graphics/blob/main/ezgif-7-182873e7ee3d.gif)
2. Long hair with global stiffness = 0.0005:
![grab-landing-page](https://github.com/loOong-Cheng/HairSimulation--Physics-based-simulation-of-computer-graphics/blob/main/ezgif-2-974bfea1b818.gif)

## References
- Müller M., Hennix B. H. M., Ratcliff J.: Position Based Dynamics. Proceedings of Virtual Reality Interactions and Physical Simulations (2006), 71–80.
- Müller M., Kim T., Chentanez N: Fast Simulation of Inextensible Hair and Fur. Workshop on Virtual Reality Interactions and Physical Simulations (2012).
- Han D., T. Harada: Real-time Hair Simulation with Efficient Hair Style Preservation. Workshop on Virtual Reality Interaction and Physical Simulation (2012).
- Petrovic L., Henne M., Aanderson J.: Volumetric Methods for Simulation and Rendering of Hair. In Tech. rep., Pixar Animation Studios (2005).

https://github.com/Wimacs/taichi_code/tree/master/hw2 (PBD)
https://github.com/clach/Realtime-Vulkan-Hair (Hair-hair interaction)
https://github.com/GPUOpen-Effects/TressFX, https://forum.taichi.graphics/t/taichi-tressfx/1453 (TressFX)
https://github.com/taichi-dev/taichi_blend (taichi_blend)

