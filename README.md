**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Grace Tan
* Tested on: Windows 11 Home 25H2 (Build 26200), AMD Ryzen 9 8945HX @ 2.50GHz, 16GB RAM, NVIDIA GeForce RTX 5060 Laptop GPU 8GB

---

# Project 1: CUDA Boids Simulation 

<p align="center">
  <img src="images/naive_boids.gif" width="800"><br>
  <sub><b>Naive Implemention</b> — 10,000 Boids</sub>
</p>
<p align="center">
  <img src="images/uniform_boids.gif" width="800"><br>
  <sub>Uniform Implementation<b></b> — 100,000 Boids</sub>
</p>
<p align="center">
  <img src="images/coherent_boids.gif" width="800"><br>
  <sub><b>Coherent Implementation</b> — 100,000 Boids</sub>
</p>

For this project, I implemented the Boids Algorithm developed by Craig Reynolds in 1986, which models flocking behavior in nature. The particles in simulation are boids, bird-like objects that are governed by three movement principles: cohesion, separation, and alignment. 

As described in Conrad Parker's notes,
1. **Cohesion** - boids try to fly towards the center of mass of neighboring boids
2. **Separation** - boids try to keep a small distance away from other boids
3. **Alignment** - boids try to match velocity with neighboring boids

To explore how different data structures and memory access patterns affect GPU performance, this project implements three algorithms with increasing levels of efficiency.
1. **Naive** - A brute-force approach that searches through all other boids to find neighbors.
2. **Uniform Grid** - A better approach utilizing the uniform grid datastructure. Boids only need to examine neighboring cells in the grid instead of naively searching through the entire space. 
3. **Coherent Grid** - An optimal approach that builds on the uniform grid by additionally sorting boid data to match grid's cell ordering. 

# Performance Analysis

Performance was measured using GLFW's elapsed-time tracking, computing the average FPS over a 10 second period. Since CUDA-GL interop calls forces synchronization between the CUDA kernel and OpenGL rendering, the CPU-side frame loop aligns with actual GPU computation. Disabling visualization removes rendering overhead so that the measured FPS more accurately reflects GPU performance. 

## For each implementation, how does changing the number of boids affect performance? Why do you think this is?

## For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

## For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

## Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

# Blooper
