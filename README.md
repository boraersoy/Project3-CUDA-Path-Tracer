CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Bora Ersoy 
  *  [https://www.linkedin.com/in/bora-ersoy-0b7950212/]()
* Tested on: 13th Gen Intel(R) Core(TM) i7-13700HX, 2100 Mhz, 16 Core(s), 24 Logical Processor(s), RTX 4060 8GB AD107

### CUDA PATH TRACER 



<img width="500" height="800" alt="cornell 2026-01-27_13-36-31z 4239samp" src="https://github.com/user-attachments/assets/5011e443-cea7-4ec1-94ea-533b080d1d6d" />
<img width="500" height="800" alt="cornell 2026-02-02_07-47-07z 4554samp" src="https://github.com/user-attachments/assets/9422f94f-3149-4105-af1a-f134b4b7109f" />



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
   - .obj model loading
   
 

 ### Specular and Diffuse Surfaces

The specular shader models perfect mirror-like reflection. Incoming rays are reflected deterministically about the surface normal, producing sharp highlights and clear reflections. No light is scattered into other directions.
The diffuse shader models rough, matte surfaces by scattering incoming light uniformly over the hemisphere around the surface normal (Lambertian reflection). New ray directions are sampled stochastically, producing soft shading and indirect illumination.
<img width="800" height="800" alt="cornell 2026-02-01_18-48-24z 4925samp" src="https://github.com/user-attachments/assets/1a224f6d-c088-4947-8cd8-95877b7c29d8" />


 ### Glass shader
The glass shader models a dielectric material by splitting energy between reflection and refraction at the surface. The Fresnel term determines the probability of reflecting versus transmitting the ray, while the transmitted ray is bent according to Snell’s law.
<img width="800" height="800" alt="cornell 2026-01-31_14-35-34z 1756samp" src="https://github.com/user-attachments/assets/86b48d8c-db6e-4fd2-ac4d-ab21f4a41f17" />

 ### Stochastic Sampled Anti Aliasing

I used supersampling with a box filter for anti-aliasing. Multiple primary rays were generated per pixel at jittered positions inside the pixel area, and their radiance values were averaged with equal weight. This corresponds to applying a box reconstruction filter, where every sample within the pixel footprint contributes uniformly to the final color. 
 
<img width="1415" height="698" alt="Screenshot 2026-01-31 174440" src="https://github.com/user-attachments/assets/9dcd4da5-34f8-4dd3-a711-c77b72118d54" />

  
<img width="1413" height="693" alt="Screenshot 2026-01-31 174454" src="https://github.com/user-attachments/assets/5a6a4b3c-1d96-42a0-9f00-bb147c6c4d85" />

### Bounding Volume Hierarchies (BVH Tree)
Although the BVH significantly reduced the number of triangle intersection tests, total frame time improved only marginally.  However traversal overhead, memory latency from pointer-heavy BVH access, and divergence from secondary rays dominated runtime. Later it can be tested on an open scene where rays terminate quicker.

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/5f24259b-2fe1-4669-9b66-bcc8d530d924" />





<img width="640" height="480" alt="bvhbrute" src="https://github.com/user-attachments/assets/e91a23de-3f79-4a2b-9348-f7d9b337a718" />


### Material Sorting 
Grouping shading work by material type (e.g., diffuse, glass, metal) so the renderer can evaluate similar BSDFs together—improving cache coherence, reducing branch divergence on GPU.
Material sorting should improve performance in theory but in small scene complexity, sorting cost shadowed the performance improvement. However in really complex scene with hunders of objects material sorting should be really helpful.
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/a560df3d-9631-4f59-9a88-4f8a9f3b6e2f" />


### First bounce caching 

Storing the results of the primary camera-ray intersection (hit point, normal, material, etc.) so you can reuse them across multiple samples—avoids re-tracing the most expensive ray and speeds up progressive rendering, debugging, or adaptive sampling

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/9c26e89e-4eaf-4974-8c9d-61f10e34edaa" />

<img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/3f0ede6f-098e-4705-88aa-2160a2a843bc" />



