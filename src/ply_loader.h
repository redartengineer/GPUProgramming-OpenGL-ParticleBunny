#ifndef PARTICLE_BUNNY_SRC_PLY_LOADER_H_
#define PARTICLE_BUNNY_SRC_PLY_LOADER_H_

#include <cstdint>
#include <string>
#include <vector>

// Stores vertex and triangle data for a 3D mesh.
struct Mesh {
  // Each vertex contains:
  // x, y, z, nx, ny, nz
  std::vector<float> vertices;

  // Three indices define one triangle.
  std::vector<std::uint32_t> indices;
};

// Loads a Stanford PLY mesh.
//
// Returns true when the file loads successfully.
bool LoadPly(const std::string& filename, Mesh* mesh);

#endif  // PARTICLE_BUNNY_SRC_PLY_LOADER_H_