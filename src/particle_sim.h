#ifndef PARTICLE_SIM_H_
#define PARTICLE_SIM_H_

#include <cstddef>

/*
 * A fur particle represents one complete hair strand.
 *
 * Each strand contains multiple points:
 *
 * root
 *  |
 *  |
 * tip
 *
 * The root remains attached to the bunny surface.
 * CUDA updates the remaining points to simulate fur.
 */
constexpr int kFurSegments = 6;


// One extra point is needed because the root is included.
constexpr int kFurPointsPerStrand =
    kFurSegments + 1;


/*
 * Initializes the CUDA fur simulation.
 *
 * The CPU provides:
 * - fur root positions
 * - surface normals
 * - strand count
 *
 * CUDA copies this data to GPU memory so
 * thousands of hairs can be updated in parallel.
 */
bool InitializeParticleSimulation(
    const float* base_positions,
    const float* normals,
    std::size_t particle_count);


/*
 * Updates fur positions for one frame.
 */
bool UpdateParticleSimulation(
    float delta_time);


/*
 * Copies updated fur strand positions back to CPU memory.
 */
bool GetParticlePositions(
    float* positions,
    std::size_t particle_count);


/*
 * Releases CUDA memory and simulation resources.
 */
void ShutdownParticleSimulation();


#endif