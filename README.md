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
<p align="center">
  <img src="images/Average FPS over 10s vs. Boid Count (128 Block Size, Visualization On).png" width="800"><br>
</p>
<p align="center">
  <img src="images/Average FPS over 10s vs. Boid Count (128 Block Size, Visualization Off).png" width="800"><br>
</p>

Generally, increasing the number of boids decreased performance across all three implementations. 

The most significant contributing factor is likely the increased neighbor count for each boid. This means that each thread must do more work processing the information of its neighbors to determine how to move in the next timestep. From the graphs, we can observe that this effect is particularly damaging for the naive approach, which completes a brute-force search through all other boids to find its neighbors. Additionally, a higher boid count increases sorting overhead, affecting the performance of the uniform grid approaches. 

## For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
<p align="center">
  <img src="images/Average FPS over 10s vs. Block Size (Boid Count 100k, Visualization On).png" width="800"><br>
</p>
<p align="center">
  <img src="images/Average FPS over 10s vs. Block Size (Boid Count 100k, Visualization Off).png" width="800"><br>
</p>

Generally, increasing the block size increased performance across all three implementations, although the uniform grid approaches showed more significant improvement over the naive method.

Notably, there are substantial improvements from increasing the block size from 16 to 32. This is likely due to threads being executed in warps of 32; any block size smaller than that would lead to wasted compute throughput since warp lanes are being underutilized. There are also significant improvements from increasing block size from 32 to 64, which could be due to removing the bottleneck potentially caused by the limit of blocks on a given streaming multiprocessor. Other improvements in performance could be attributed reduced block count, which decreases the overhead associated with launching and scheduling multiple blocks. 

## For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
For all three implementations, there is a notable performance improvement using coherent grid over uniform grid.

I was initially unsure of why the coherent grid approach performed so much better than the uniform grid approach; although coherent grid removes the additional step of referencing the array of boid pointers (a global memory read), it also requires device memory allocation for two additional buffers. Then, I realized that storing the positions and velocities of boids within the same cell results in memory coalescing, allowing warps to make bigger memory requests since requested values are stored close together in memory. 

## Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
I kept block size at 128 and boid count at 100k: 

<div align="center">
  <table>
    <tr>
      <th colspan="3">Visualization On</th>
    </tr>
    <tr>
      <th></th>
      <th>Uniform</th>
      <th>Coherent</th>
    </tr>
    <tr>
      <td>8 Cells, 2x Max Rule Distance</td>
      <td>438.437321</td>
      <td>561.076609</td>
    </tr>
    <tr>
      <td>27 Cells, 1x Max Rule Distance</td>
      <td>589.765764</td>
      <td>641.464905</td>
    </tr>
    <tr>
      <th colspan="3">Visualization Off</th>
    </tr>
    <tr>
      <th></th>
      <th>Uniform</th>
      <th>Coherent</th>
    </tr>
    <tr>
      <td>8 Cells, 2x Max Rule Distance</td>
      <td>680.343491</td>
      <td>994.047743</td>
    </tr>
    <tr>
      <td>27 Cells, 1x Max Rule Distance</td>
      <td>905.059607</td>
      <td>1312.235267</td>
    </tr>
  </table>
</div>

Yes, there is an improvement across both methods and with visualization both on and off. This could be due to the simpler 27 neighboring cells check, which does not require additional calculations to determine which cells might contain neighbors. The bigger cell size in the 8 cell check could also result in boids that are substantially further than the largest rule distance being checked in the loop. The reduced cell size could also mean there are more empty cells that the loop can simply skip over. 
  
# Blooper
