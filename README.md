CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Bora Ersoy 
  *  [https://www.linkedin.com/in/bora-ersoy-0b7950212/]()
* Tested on: 13th Gen Intel(R) Core(TM) i7-13700HX, 2100 Mhz, 16 Core(s), 24 Logical Processor(s), RTX 4060 8GB AD107

### CUDA PATH TRACER 



<img width="600" height="800" alt="cornell 2026-01-27_13-36-31z 4239samp" src="https://github.com/user-attachments/assets/5011e443-cea7-4ec1-94ea-533b080d1d6d" />
<img width="600" height="800" alt="cornell 2026-01-31_16-51-26z 1800samp" src="https://github.com/user-attachments/assets/bed4a306-9bb4-43dc-952c-b9e8b0d7df6d" />


### Features

 CUDA-based path tracer capable of rendering globally-illuminated images 

- Visual Features
 - Perfect specular and diffuse surfaces
 - Glass shader (refraction)
 - Stochastic Sampled Anti Aliasing
- Performance Optimizations
  - BVH
  - Material Sorting
  - First bounce caching
  - Ray Termination with Stream Compaction
 - Model loading
   -.obj model loading
   
 

 ### Specular and Diffuse Surfaces

 Sphere has specular and bunny object has diffuse surface.

 ### Glass shader
 Glass surface using refraction.
<img width="800" height="800" alt="cornell 2026-01-31_14-35-34z 1756samp" src="https://github.com/user-attachments/assets/86b48d8c-db6e-4fd2-ac4d-ab21f4a41f17" />

 ### Stochastic Sampled Anti Aliasing

 Here is the difference between anti aliased version of the image. Edges are sharper in the reflection with box filter anti alising.
 
<img width="1415" height="698" alt="Screenshot 2026-01-31 174440" src="https://github.com/user-attachments/assets/9dcd4da5-34f8-4dd3-a711-c77b72118d54" />

  
<img width="1413" height="693" alt="Screenshot 2026-01-31 174454" src="https://github.com/user-attachments/assets/5a6a4b3c-1d96-42a0-9f00-bb147c6c4d85" />

### Bounding Volume Hierarchies (BVH Tree)
Although the BVH significantly reduced the number of triangle intersection tests, total frame time improved only marginally.  However traversal overhead, memory latency from pointer-heavy BVH access, and divergence from secondary rays dominated runtime. Later it can be tested on an open scene where rays terminate quicker.
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/13467071-fdd8-4c22-a08a-dd76ec9f6bf3" />





<img width="640" height="480" alt="bvhbrute" src="https://github.com/user-attachments/assets/e91a23de-3f79-4a2b-9348-f7d9b337a718" />


### Material Sorting 
Grouping shading work by material type (e.g., diffuse, glass, metal) so the renderer can evaluate similar BSDFs together—improving cache coherence, reducing branch divergence on GPU.
Material sorting should improve performance in theory but in small scene complexity, sorting cost shadowed the performance improvement. However in really complex scene with hunders of objects material sorting should be really helpful.
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/2a97ed29-7bad-4e54-b2f5-19b81c284a6f" />


### First bounce caching 

Storing the results of the primary camera-ray intersection (hit point, normal, material, etc.) so you can reuse them across multiple samples—avoids re-tracing the most expensive ray and speeds up progressive rendering, debugging, or adaptive sampling

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/9c26e89e-4eaf-4974-8c9d-61f10e34edaa" />

<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/3f0ede6f-098e-4705-88aa-2160a2a843bc" />



