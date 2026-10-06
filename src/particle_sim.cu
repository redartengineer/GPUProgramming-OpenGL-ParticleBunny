#include "particle_sim.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>

namespace {

constexpr int kThreadsPerBlock = 256;

/*
 * Hair length.
 *
 * Your bunny is normalized to approximately
 * one world-space unit, so this is deliberately
 * short.
 */
constexpr float kHairLength = 0.014f;

/*
 * Wind strength.
 *
 * The wind is strongest toward the tip.
 */
constexpr float kWindStrength = 0.0040f;

/*
 * Wind frequency.
 */
constexpr float kWindFrequency = 1.8f;

/*
 * Small random-looking variation in hair length.
 */
constexpr float kLengthVariation = 0.20f;

/*
 * CUDA device arrays.
 */
/*
 * GPU memory used by the fur simulation.
 *
 * g_base_positions:
 *   Original fur root positions generated from the bunny surface.
 *
 * g_normals:
 *   Surface directions used to grow each hair away from the mesh.
 *
 * g_device_positions:
 *   Animated strand points updated every frame by CUDA.
 */
float* g_base_positions = nullptr;

float* g_normals = nullptr;

float* g_device_positions = nullptr;

std::size_t g_particle_count = 0;

float g_time = 0.0f;

bool g_initialized = false;


// CUDA error helper

bool CheckCuda(
    cudaError_t error,
    const char* operation) {

    if (error != cudaSuccess) {

        std::fprintf(
            stderr,
            "CUDA error during %s: %s\n",
            operation,
            cudaGetErrorString(error)
        );

        return false;
    }

    return true;
}


// Fur strand kernel One CUDA thread controls one entire hair. The root is fixed to the bunny surface. Every additional point is placed farther away from the surface along the normal. Wind bends the strand progressively more toward its tip.

__global__ void UpdateFurKernel(
    const float* base_positions,
    const float* normals,
    float* positions,
    std::size_t particle_count,
    float time) {

    const std::size_t strand =
        static_cast<std::size_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;

    if (strand >= particle_count) {
        return;
    }


    /*
     * Three floats per surface position.
     */
    const std::size_t base =
        strand * 3;


    /*
     * Seven points per strand.
     */
    const std::size_t output_base =
        strand *
        kFurPointsPerStrand *
        3;


    // Bunny surface position

    const float px =
        base_positions[base + 0];

    const float py =
        base_positions[base + 1];

    const float pz =
        base_positions[base + 2];


    // Surface normal

    float nx =
        normals[base + 0];

    float ny =
        normals[base + 1];

    float nz =
        normals[base + 2];


    /*
     * Normalize normal.
     */

    const float normal_length =
        sqrtf(
            nx * nx +
            ny * ny +
            nz * nz
        );

    if (normal_length > 0.000001f) {

        nx /= normal_length;
        ny /= normal_length;
        nz /= normal_length;

    } else {

        nx = 0.0f;
        ny = 1.0f;
        nz = 0.0f;
    }

    // Per-strand variation creates deterministic differences without using random numbers.
    const float variation =
        sinf(px * 127.1f + py * 311.7f + pz * 74.3f);

    const float normalized_variation =
        0.5f + 0.5f * variation;

    const float strand_length =
        kHairLength *
        (1.0f + (normalized_variation - 0.5f) * kLengthVariation);

    // Spatial wind phase.

    const float phase =
        time * kWindFrequency +
        px * 19.0f +
        py * 11.0f +
        pz * 17.0f;


    /*
     * Wind direction.
     *
     * Different locations on the bunny have
     * slightly different motion.
     */

    const float wind_x =
        sinf(phase) *
        kWindStrength;

    const float wind_z =
        cosf(
            phase * 0.73f
        ) *
        kWindStrength;

    const float wind_y =
        sinf(
            phase * 0.51f
        ) *
        kWindStrength *
        0.35f;


    // Build every point in the strand.

    for (
        int point = 0;
        point < kFurPointsPerStrand;
        ++point) {

        /*
         * t = 0 at root
         * t = 1 at tip
         */

        const float t =
            static_cast<float>(point) /
            static_cast<float>(kFurSegments);


        /*
         * Hair naturally curves slightly
         * rather than remaining perfectly straight.
         *
         * t^2 means the root remains stable while
         * the tip bends more.
         */

        const float bend =
            t * t;


        /*
         * Distance away from bunny.
         */

        const float distance =
            strand_length * t;


        /*
         * Start from the bunny surface and
         * travel along the surface normal.
         */

        float x =
            px +
            nx * distance;

        float y =
            py +
            ny * distance;

        float z =
            pz +
            nz * distance;


        /*
         * Apply wind progressively toward
         * the tip.
         *
         * The root receives zero wind.
         */

        x +=
            wind_x *
            bend;

        y +=
            wind_y *
            bend;

        z +=
            wind_z *
            bend;


        /*
         * Add a tiny side-to-side curl.
         *
         * This prevents the fur from looking
         * like thousands of perfectly identical
         * straight needles.
         */

        const float curl =
            sinf(
                phase +
                t * 4.5f
            ) *
            0.0012f *
            bend;


        x +=
            curl *
            (1.0f - fabsf(nx));

        z +=
            curl *
            (1.0f - fabsf(nz));


        // Store point.

        const std::size_t output =
            output_base +
            static_cast<std::size_t>(point) * 3;

        positions[output + 0] = x;
        positions[output + 1] = y;
        positions[output + 2] = z;
    }
}

}  // namespace


// / INITIALIZATION /

bool InitializeParticleSimulation(
    const float* base_positions,
    const float* normals,
    std::size_t particle_count) {

    if (
        base_positions == nullptr ||
        normals == nullptr ||
        particle_count == 0) {

        std::fprintf(
            stderr,
            "Invalid fur particle data.\n"
        );

        return false;
    }


    /*
     * Make sure old allocations are gone.
     */

    ShutdownParticleSimulation();


    g_particle_count =
        particle_count;


    /*
     * Surface arrays:
     *
     * particle_count * xyz
     */

    const std::size_t surface_size =
        particle_count *
        3 *
        sizeof(float);


    /*
     * Strand output:
     *
     * particle_count *
     * points per strand *
     * xyz
     */

    const std::size_t strand_size =
        particle_count *
        kFurPointsPerStrand *
        3 *
        sizeof(float);


    // Base positions

    if (
        !CheckCuda(
            cudaMalloc(
                reinterpret_cast<void**>(
                    &g_base_positions
                ),
                surface_size
            ),
            "cudaMalloc base positions"
        )
    ) {

        ShutdownParticleSimulation();

        return false;
    }


    // Normals

    if (
        !CheckCuda(
            cudaMalloc(
                reinterpret_cast<void**>(
                    &g_normals
                ),
                surface_size
            ),
            "cudaMalloc normals"
        )
    ) {

        ShutdownParticleSimulation();

        return false;
    }


    // Strand positions

    if (
        !CheckCuda(
            cudaMalloc(
                reinterpret_cast<void**>(
                    &g_device_positions
                ),
                strand_size
            ),
            "cudaMalloc fur strand positions"
        )
    ) {

        ShutdownParticleSimulation();

        return false;
    }


    /*
     * Copy surface positions.
     */

    if (
        !CheckCuda(
            cudaMemcpy(
                g_base_positions,
                base_positions,
                surface_size,
                cudaMemcpyHostToDevice
            ),
            "copy base positions"
        )
    ) {

        ShutdownParticleSimulation();

        return false;
    }


    /*
     * Copy surface normals.
     */

    if (
        !CheckCuda(
            cudaMemcpy(
                g_normals,
                normals,
                surface_size,
                cudaMemcpyHostToDevice
            ),
            "copy normals"
        )
    ) {

        ShutdownParticleSimulation();

        return false;
    }


    /*
     * Initialize strand positions.
     *
     * They will be overwritten by the first
     * kernel update.
     */

    if (
        !CheckCuda(
            cudaMemset(
                g_device_positions,
                0,
                strand_size
            ),
            "initialize fur positions"
        )
    ) {

        ShutdownParticleSimulation();

        return false;
    }


    g_time =
        0.0f;

    g_initialized =
        true;


    // CUDA device information

    int device_count = 0;

    if (
        CheckCuda(
            cudaGetDeviceCount(
                &device_count
            ),
            "cudaGetDeviceCount"
        )
    ) {

        if (device_count > 0) {

            int current_device = 0;

            CheckCuda(
                cudaGetDevice(
                    &current_device
                ),
                "cudaGetDevice"
            );

            cudaDeviceProp properties{};

            if (
                CheckCuda(
                    cudaGetDeviceProperties(
                        &properties,
                        current_device
                    ),
                    "cudaGetDeviceProperties"
                )
            ) {

                std::printf(
                    "CUDA fur simulation initialized.\n"
                );

                std::printf(
                    "CUDA device: %s\n",
                    properties.name
                );

                std::printf(
                    "Fur strands: %zu\n",
                    particle_count
                );

                std::printf(
                    "Points per strand: %d\n",
                    kFurPointsPerStrand
                );

                std::printf(
                    "Total fur points: %zu\n",
                    particle_count *
                    static_cast<std::size_t>(
                        kFurPointsPerStrand
                    )
                );

                std::printf(
                    "Compute capability: %d.%d\n",
                    properties.major,
                    properties.minor
                );
            }
        }
    }


    return true;
}


// / UPDATE /

bool UpdateParticleSimulation(
    float delta_time) {

    if (!g_initialized) {
        return false;
    }


    if (delta_time < 0.0f) {
        delta_time = 0.0f;
    }


    if (delta_time > 0.05f) {
        delta_time = 0.05f;
    }


    g_time +=
        delta_time;


    /*
     * One CUDA thread per hair.
     */

    const int blocks =
        static_cast<int>(
            (
                g_particle_count +
                kThreadsPerBlock -
                1
            ) /
            kThreadsPerBlock
        );


    UpdateFurKernel<<<
        blocks,
        kThreadsPerBlock
    >>>(
        g_base_positions,
        g_normals,
        g_device_positions,
        g_particle_count,
        g_time
    );


    /*
     * Check kernel launch.
     */

    const cudaError_t launch_error =
        cudaGetLastError();

    if (
        launch_error != cudaSuccess) {

        std::fprintf(
            stderr,
            "CUDA error during fur kernel launch: %s\n",
            cudaGetErrorString(
                launch_error
            )
        );

        return false;
    }


    /*
     * Wait for completion before copying
     * the strand data to the CPU.
     */

    const cudaError_t sync_error =
        cudaDeviceSynchronize();

    if (
        sync_error != cudaSuccess) {

        std::fprintf(
            stderr,
            "CUDA error during fur kernel execution: %s\n",
            cudaGetErrorString(
                sync_error
            )
        );

        return false;
    }


    return true;
}


// / GET STRAND POSITIONS /

bool GetParticlePositions(
    float* positions,
    std::size_t particle_count) {

    if (
        !g_initialized ||
        positions == nullptr ||
        particle_count != g_particle_count) {

        return false;
    }


    const std::size_t size =
        g_particle_count *
        kFurPointsPerStrand *
        3 *
        sizeof(float);


    return CheckCuda(
        cudaMemcpy(
            positions,
            g_device_positions,
            size,
            cudaMemcpyDeviceToHost
        ),
        "copy fur strand positions"
    );
}


// / SHUTDOWN /

void ShutdownParticleSimulation() {

    if (g_device_positions != nullptr) {

        cudaFree(
            g_device_positions
        );

        g_device_positions =
            nullptr;
    }


    if (g_normals != nullptr) {

        cudaFree(
            g_normals
        );

        g_normals =
            nullptr;
    }


    if (g_base_positions != nullptr) {

        cudaFree(
            g_base_positions
        );

        g_base_positions =
            nullptr;
    }


    g_particle_count =
        0;

    g_time =
        0.0f;

    g_initialized =
        false;
}