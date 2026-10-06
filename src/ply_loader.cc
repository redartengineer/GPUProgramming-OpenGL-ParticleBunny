#include "ply_loader.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Vec3 {
  float x;
  float y;
  float z;
};

Vec3 Subtract(const Vec3& a, const Vec3& b) {
  return {
      a.x - b.x,
      a.y - b.y,
      a.z - b.z,
  };
}

Vec3 Cross(const Vec3& a, const Vec3& b) {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

void Normalize(Vec3* vector) {
  const float length = std::sqrt(
      vector->x * vector->x +
      vector->y * vector->y +
      vector->z * vector->z);

  if (length > 0.000001f) {
    vector->x /= length;
    vector->y /= length;
    vector->z /= length;
  }
}

// Centers the model at the origin and scales its largest dimension to 1.0.
bool NormalizePositions(std::vector<Vec3>* positions) {
  if (positions == nullptr || positions->empty()) {
    return false;
  }

  Vec3 minimum = positions->front();
  Vec3 maximum = positions->front();

  // Find the model's bounding box.
  for (const Vec3& position : *positions) {
    minimum.x = std::min(minimum.x, position.x);
    minimum.y = std::min(minimum.y, position.y);
    minimum.z = std::min(minimum.z, position.z);

    maximum.x = std::max(maximum.x, position.x);
    maximum.y = std::max(maximum.y, position.y);
    maximum.z = std::max(maximum.z, position.z);
  }

  const Vec3 center = {
      (minimum.x + maximum.x) * 0.5f,
      (minimum.y + maximum.y) * 0.5f,
      (minimum.z + maximum.z) * 0.5f,
  };

  const float width = maximum.x - minimum.x;
  const float height = maximum.y - minimum.y;
  const float depth = maximum.z - minimum.z;

  const float largest_dimension =
      std::max({width, height, depth});

  if (largest_dimension <= 0.000001f) {
    std::cerr << "Model has an invalid bounding box.\n";
    return false;
  }

  // Center and normalize every vertex.
  for (Vec3& position : *positions) {
    position.x =
        (position.x - center.x) / largest_dimension;
    position.y =
        (position.y - center.y) / largest_dimension;
    position.z =
        (position.z - center.z) / largest_dimension;
  }

  std::cout << "Original bounding box:\n"
            << "  Minimum: "
            << minimum.x << ", "
            << minimum.y << ", "
            << minimum.z << '\n'
            << "  Maximum: "
            << maximum.x << ", "
            << maximum.y << ", "
            << maximum.z << '\n'
            << "Largest dimension: "
            << largest_dimension << '\n';

  return true;
}

}  // namespace

bool LoadPly(const std::string& filename, Mesh* mesh) {
  if (mesh == nullptr) {
    std::cerr << "Mesh pointer is null.\n";
    return false;
  }

  std::ifstream file(filename);

  if (!file) {
    std::cerr << "Could not open PLY file: " << filename << '\n';
    return false;
  }

  std::string line;
  std::size_t vertex_count = 0;
  std::size_t face_count = 0;
  bool is_ascii = false;

  // Read the PLY header.
  while (std::getline(file, line)) {
    if (line == "format ascii 1.0") {
      is_ascii = true;
    } else if (line.rfind("element vertex ", 0) == 0) {
      std::istringstream stream(line);
      std::string element;
      std::string vertex;

      stream >> element >> vertex >> vertex_count;
    } else if (line.rfind("element face ", 0) == 0) {
      std::istringstream stream(line);
      std::string element;
      std::string face;

      stream >> element >> face >> face_count;
    } else if (line == "end_header") {
      break;
    }
  }

  if (!is_ascii) {
    std::cerr
        << "The current loader supports ASCII PLY files only.\n";
    return false;
  }

  if (vertex_count == 0 || face_count == 0) {
    std::cerr << "PLY file does not contain valid mesh data.\n";
    return false;
  }

  std::vector<Vec3> positions(vertex_count);
  std::vector<Vec3> normals(
      vertex_count, Vec3{0.0f, 0.0f, 0.0f});

  // Read vertex positions.
  for (std::size_t i = 0; i < vertex_count; ++i) {
    if (!std::getline(file, line)) {
      std::cerr << "Unexpected end of vertex data.\n";
      return false;
    }

    std::istringstream stream(line);

    if (!(stream >> positions[i].x
                 >> positions[i].y
                 >> positions[i].z)) {
      std::cerr << "Failed to read vertex " << i << ".\n";
      return false;
    }
  }

  // Center the model and normalize its scale.
  if (!NormalizePositions(&positions)) {
    return false;
  }

  mesh->indices.clear();

  // Read faces and convert them into triangles.
  for (std::size_t i = 0; i < face_count; ++i) {
    if (!std::getline(file, line)) {
      std::cerr << "Unexpected end of face data.\n";
      return false;
    }

    std::istringstream stream(line);
    int vertices_per_face = 0;

    if (!(stream >> vertices_per_face)) {
      std::cerr << "Failed to read face " << i << ".\n";
      return false;
    }

    if (vertices_per_face < 3) {
      continue;
    }

    std::vector<std::uint32_t> face(
        static_cast<std::size_t>(vertices_per_face));

    for (int j = 0; j < vertices_per_face; ++j) {
      if (!(stream >> face[static_cast<std::size_t>(j)])) {
        std::cerr << "Failed to read face indices.\n";
        return false;
      }
    }

    // Convert polygon faces into triangles using a triangle fan.
    for (int j = 1; j < vertices_per_face - 1; ++j) {
      mesh->indices.push_back(face[0]);
      mesh->indices.push_back(face[j]);
      mesh->indices.push_back(face[j + 1]);
    }
  }

  // Calculate smooth vertex normals.
  for (std::size_t i = 0; i + 2 < mesh->indices.size(); i += 3) {
    const std::uint32_t index_0 = mesh->indices[i];
    const std::uint32_t index_1 = mesh->indices[i + 1];
    const std::uint32_t index_2 = mesh->indices[i + 2];

    if (index_0 >= positions.size() ||
        index_1 >= positions.size() ||
        index_2 >= positions.size()) {
      std::cerr << "PLY file contains an invalid vertex index.\n";
      return false;
    }

    const Vec3 edge_1 =
        Subtract(positions[index_1], positions[index_0]);

    const Vec3 edge_2 =
        Subtract(positions[index_2], positions[index_0]);

    const Vec3 face_normal = Cross(edge_1, edge_2);

    normals[index_0].x += face_normal.x;
    normals[index_0].y += face_normal.y;
    normals[index_0].z += face_normal.z;

    normals[index_1].x += face_normal.x;
    normals[index_1].y += face_normal.y;
    normals[index_1].z += face_normal.z;

    normals[index_2].x += face_normal.x;
    normals[index_2].y += face_normal.y;
    normals[index_2].z += face_normal.z;
  }

  for (Vec3& normal : normals) {
    Normalize(&normal);
  }

  mesh->vertices.clear();
  mesh->vertices.reserve(vertex_count * 6);

  // Store position and normal data for OpenGL.
  for (std::size_t i = 0; i < vertex_count; ++i) {
    mesh->vertices.push_back(positions[i].x);
    mesh->vertices.push_back(positions[i].y);
    mesh->vertices.push_back(positions[i].z);

    mesh->vertices.push_back(normals[i].x);
    mesh->vertices.push_back(normals[i].y);
    mesh->vertices.push_back(normals[i].z);
  }

  std::cout << "Loaded mesh successfully.\n"
            << "Vertices:  " << vertex_count << '\n'
            << "Triangles: " << mesh->indices.size() / 3 << '\n';

  return true;
}