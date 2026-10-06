#include <glad/gl.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "particle_sim.h"
#include "ply_loader.h"

namespace {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;

constexpr char kWindowTitle[] =
    "Dandelion Bunny";

constexpr int kDefaultParticleCount =
    180000;

constexpr int kMaximumParticleCount =
    300000;


/*
 * The CUDA simulation defines the actual number of
 * points per strand.
 *
 * We use a separate display-side name here so that
 * there is no conflict with the CUDA header.
 */
constexpr int kDisplayPointsPerStrand =
    7;


// CAMERA

struct CameraState {

  float distance =
      3.0f;

  float height =
      0.20f;

  float horizontal =
      0.0f;

  float target_y =
      0.0f;
};


// PROGRAM OPTIONS

struct ProgramOptions {

  std::string model_path =
      "Assets/bunny/reconstruction/bun_zipper.ply";

  int particle_count =
      kDefaultParticleCount;

  bool show_help =
      false;
};


// FILE UTILITIES

std::string ReadTextFile(
    const std::string& filename) {

  std::ifstream file(
      filename);

  if (!file) {

    std::cerr
        << "Could not open file: "
        << filename
        << '\n';

    return {};
  }

  std::stringstream buffer;

  buffer
      << file.rdbuf();

  return buffer.str();
}


// SHADER COMPILATION

GLuint CompileShader(
    GLenum type,
    const std::string& source) {

  const GLuint shader =
      glCreateShader(type);

  const char* source_pointer =
      source.c_str();

  glShaderSource(
      shader,
      1,
      &source_pointer,
      nullptr);

  glCompileShader(
      shader);

  GLint success =
      GL_FALSE;

  glGetShaderiv(
      shader,
      GL_COMPILE_STATUS,
      &success);

  if (success != GL_TRUE) {

    char log[4096];

    glGetShaderInfoLog(
        shader,
        sizeof(log),
        nullptr,
        log);

    std::cerr
        << "Shader compilation failed:\n"
        << log
        << '\n';

    glDeleteShader(
        shader);

    return 0;
  }

  return shader;
}


GLuint CreateShaderProgram(
    const std::string& vertex_path,
    const std::string& fragment_path) {

  const std::string vertex_source =
      ReadTextFile(
          vertex_path);

  const std::string fragment_source =
      ReadTextFile(
          fragment_path);

  if (vertex_source.empty() ||
      fragment_source.empty()) {

    return 0;
  }

  const GLuint vertex_shader =
      CompileShader(
          GL_VERTEX_SHADER,
          vertex_source);

  if (vertex_shader == 0) {

    return 0;
  }

  const GLuint fragment_shader =
      CompileShader(
          GL_FRAGMENT_SHADER,
          fragment_source);

  if (fragment_shader == 0) {

    glDeleteShader(
        vertex_shader);

    return 0;
  }

  const GLuint program =
      glCreateProgram();

  glAttachShader(
      program,
      vertex_shader);

  glAttachShader(
      program,
      fragment_shader);

  glLinkProgram(
      program);

  GLint success =
      GL_FALSE;

  glGetProgramiv(
      program,
      GL_LINK_STATUS,
      &success);

  glDeleteShader(
      vertex_shader);

  glDeleteShader(
      fragment_shader);

  if (success != GL_TRUE) {

    char log[4096];

    glGetProgramInfoLog(
        program,
        sizeof(log),
        nullptr,
        log);

    std::cerr
        << "Shader program link failed:\n"
        << log
        << '\n';

    glDeleteProgram(
        program);

    return 0;
  }

  return program;
}


// WINDOW CALLBACK

void FramebufferSizeCallback(
    GLFWwindow*,
    int width,
    int height) {

  glViewport(
      0,
      0,
      width,
      height);
}


// MOUSE WHEEL

void ScrollCallback(
    GLFWwindow* window,
    double,
    double y_offset) {

  CameraState* camera =
      static_cast<CameraState*>(
          glfwGetWindowUserPointer(
              window));

  if (camera == nullptr) {

    return;
  }

  camera->distance -=
      static_cast<float>(
          y_offset) *
      0.20f;

  camera->distance =
      std::clamp(
          camera->distance,
          1.0f,
          8.0f);
}


// HELP

void PrintHelp() {

  std::cout
      << "Dandelion Bunny\n"
      << "----------------\n\n"

      << "Usage:\n"
      << "  ParticleBunny.exe [options]\n\n"

      << "Options:\n"

      << "  --model <path>\n"
      << "      Path to an ASCII PLY model.\n\n"

      << "  --particles <count>\n"
      << "      Number of fur strands.\n\n"

      << "  --help\n"
      << "      Show this help message.\n\n"

      << "Simulation controls:\n"
      << "  W / S       Move camera up / down\n"
      << "  A / D       Move camera left / right\n"
      << "  + / -       Zoom in / out\n"
      << "  Mouse wheel Zoom in / out\n"
      << "  R           Reset camera\n"
      << "  ESC         Quit\n";
}


// ARGUMENT PARSING

bool ParseArguments(
    int argc,
    char* argv[],
    ProgramOptions* options) {

  if (options == nullptr) {

    return false;
  }

  for (int i = 1;
       i < argc;
       ++i) {

    const std::string argument =
        argv[i];


    if (argument == "--model") {

      if (i + 1 >= argc) {

        std::cerr
            << "--model requires a path.\n";

        return false;
      }

      options->model_path =
          argv[++i];
    }


    else if (
        argument == "--particles") {

      if (i + 1 >= argc) {

        std::cerr
            << "--particles requires a number.\n";

        return false;
      }

      try {

        options->particle_count =
            std::stoi(
                argv[++i]);

      }

      catch (...) {

        std::cerr
            << "Particle count must be "
            << "an integer.\n";

        return false;
      }


      if (options->particle_count <= 0 ||
          options->particle_count >
              kMaximumParticleCount) {

        std::cerr
            << "Particle count must be "
            << "between 1 and "
            << kMaximumParticleCount
            << ".\n";

        return false;
      }
    }


    else if (
        argument == "--help") {

      options->show_help =
          true;

      return true;
    }


    else {

      std::cerr
          << "Unknown argument: "
          << argument
          << '\n';

      return false;
    }
  }

  return true;
}


// VECTOR UTILITIES

struct Vec3 {

  float x;
  float y;
  float z;
};


Vec3 PositionAt(
    const Mesh& mesh,
    std::uint32_t index) {

  const std::size_t offset =
      static_cast<std::size_t>(
          index) *
      6;

  return {

      mesh.vertices[
          offset + 0],

      mesh.vertices[
          offset + 1],

      mesh.vertices[
          offset + 2]
  };
}


Vec3 NormalAt(
    const Mesh& mesh,
    std::uint32_t index) {

  const std::size_t offset =
      static_cast<std::size_t>(
          index) *
      6;

  return {

      mesh.vertices[
          offset + 3],

      mesh.vertices[
          offset + 4],

      mesh.vertices[
          offset + 5]
  };
}


Vec3 Normalize(
    Vec3 value) {

  const float length =
      std::sqrt(

          value.x * value.x +
          value.y * value.y +
          value.z * value.z
      );


  if (length >
      0.000001f) {

    value.x /=
        length;

    value.y /=
        length;

    value.z /=
        length;
  }

  else {

    value = {

        0.0f,
        1.0f,
        0.0f
    };
  }


  return value;
}


Vec3 LerpBarycentric(
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    float u,
    float v) {

  const float w =
      1.0f -
      u -
      v;

  return {

      a.x * w +
          b.x * u +
          c.x * v,

      a.y * w +
          b.y * u +
          c.y * v,

      a.z * w +
          b.z * u +
          c.z * v
  };
}


float TriangleArea(
    const Vec3& a,
    const Vec3& b,
    const Vec3& c) {

  const float abx =
      b.x - a.x;

  const float aby =
      b.y - a.y;

  const float abz =
      b.z - a.z;


  const float acx =
      c.x - a.x;

  const float acy =
      c.y - a.y;

  const float acz =
      c.z - a.z;


  const float cross_x =
      aby * acz -
      abz * acy;

  const float cross_y =
      abz * acx -
      abx * acz;

  const float cross_z =
      abx * acy -
      aby * acx;


  return
      0.5f *
      std::sqrt(

          cross_x * cross_x +
          cross_y * cross_y +
          cross_z * cross_z
      );
}


// FUR ROOT GENERATION

void CreateFurData(
    const Mesh& bunny,
    int requested_particle_count,
    std::vector<float>* base_positions,
    std::vector<float>* normals) {

  base_positions->clear();

  normals->clear();


  const std::size_t triangle_count =
      bunny.indices.size() /
      3;


  if (triangle_count == 0) {

    return;
  }


  std::vector<float>
      cumulative_area(
          triangle_count);


  float total_area =
      0.0f;


  for (std::size_t i = 0;
       i < triangle_count;
       ++i) {

    const std::uint32_t ia =
        bunny.indices[
            i * 3 + 0];

    const std::uint32_t ib =
        bunny.indices[
            i * 3 + 1];

    const std::uint32_t ic =
        bunny.indices[
            i * 3 + 2];


    const Vec3 a =
        PositionAt(
            bunny,
            ia);

    const Vec3 b =
        PositionAt(
            bunny,
            ib);

    const Vec3 c =
        PositionAt(
            bunny,
            ic);


    total_area +=
        TriangleArea(
            a,
            b,
            c);


    cumulative_area[i] =
        total_area;
  }


  if (total_area <= 0.0f) {

    return;
  }


  base_positions->reserve(

      static_cast<std::size_t>(
          requested_particle_count) *
      3
  );


  normals->reserve(

      static_cast<std::size_t>(
          requested_particle_count) *
      3
  );


  std::mt19937 generator(
      12345);


  std::uniform_real_distribution<float>
      area_distribution(
          0.0f,
          total_area);


  std::uniform_real_distribution<float>
      unit_distribution(
          0.0f,
          1.0f);


  for (int particle = 0;
       particle <
           requested_particle_count;
       ++particle) {

    const float target_area =
        area_distribution(
            generator);


    const auto triangle_it =
        std::lower_bound(

            cumulative_area.begin(),

            cumulative_area.end(),

            target_area
        );


    std::size_t triangle =
        static_cast<std::size_t>(

            std::distance(

                cumulative_area.begin(),

                triangle_it
            )
        );


    if (triangle >=
        triangle_count) {

      triangle =
          triangle_count - 1;
    }


    const std::uint32_t ia =
        bunny.indices[
            triangle * 3 + 0];

    const std::uint32_t ib =
        bunny.indices[
            triangle * 3 + 1];

    const std::uint32_t ic =
        bunny.indices[
            triangle * 3 + 2];


    const Vec3 a =
        PositionAt(
            bunny,
            ia);

    const Vec3 b =
        PositionAt(
            bunny,
            ib);

    const Vec3 c =
        PositionAt(
            bunny,
            ic);


    const Vec3 na =
        NormalAt(
            bunny,
            ia);

    const Vec3 nb =
        NormalAt(
            bunny,
            ib);

    const Vec3 nc =
        NormalAt(
            bunny,
            ic);


    float u =
        unit_distribution(
            generator);

    float v =
        unit_distribution(
            generator);


    if (u + v > 1.0f) {

      u =
          1.0f - u;

      v =
          1.0f - v;
    }


    const Vec3 position =
        LerpBarycentric(

            a,
            b,
            c,

            u,
            v
        );


    const Vec3 normal =
        Normalize(

            LerpBarycentric(

                na,
                nb,
                nc,

                u,
                v
            )
        );


    base_positions->push_back(
        position.x);

    base_positions->push_back(
        position.y);

    base_positions->push_back(
        position.z);


    normals->push_back(
        normal.x);

    normals->push_back(
        normal.y);

    normals->push_back(
        normal.z);
  }
}


// MAIN

}  // namespace


int main(
    int argc,
    char* argv[]) {

  ProgramOptions options;


  // Arguments

  if (!ParseArguments(
          argc,
          argv,
          &options)) {

    return 1;
  }


  if (options.show_help) {

    PrintHelp();

    return 0;
  }


  // GLFW

  if (glfwInit() !=
      GLFW_TRUE) {

    std::cerr
        << "Failed to initialize GLFW.\n";

    return 1;
  }


  glfwWindowHint(
      GLFW_CONTEXT_VERSION_MAJOR,
      4);

  glfwWindowHint(
      GLFW_CONTEXT_VERSION_MINOR,
      3);

  glfwWindowHint(
      GLFW_OPENGL_PROFILE,
      GLFW_OPENGL_CORE_PROFILE);


  GLFWwindow* window =
      glfwCreateWindow(

          kWindowWidth,
          kWindowHeight,

          kWindowTitle,

          nullptr,
          nullptr
      );


  if (window == nullptr) {

    std::cerr
        << "Failed to create the "
        << "OpenGL window.\n";

    glfwTerminate();

    return 1;
  }


  glfwMakeContextCurrent(
      window);


  // Camera

  CameraState camera;


  glfwSetWindowUserPointer(
      window,
      &camera);


  glfwSetScrollCallback(
      window,
      ScrollCallback);


  glfwSetFramebufferSizeCallback(
      window,
      FramebufferSizeCallback);


  // GLAD

  const int version =
      gladLoadGL(
          glfwGetProcAddress);


  if (version == 0) {

    std::cerr
        << "Failed to initialize GLAD.\n";

    glfwDestroyWindow(
        window);

    glfwTerminate();

    return 1;
  }


  // OpenGL information

  const char* vendor =
      reinterpret_cast<
          const char*>(
          glGetString(
              GL_VENDOR));


  const char* renderer =
      reinterpret_cast<
          const char*>(
          glGetString(
              GL_RENDERER));


  const char* gl_version =
      reinterpret_cast<
          const char*>(
          glGetString(
              GL_VERSION));


  std::cout

      << "Dandelion Bunny\n"
      << "----------------\n"

      << "OpenGL: "
      << GLAD_VERSION_MAJOR(version)
      << '.'
      << GLAD_VERSION_MINOR(version)
      << '\n'

      << "OpenGL Vendor: "
      << (vendor != nullptr
              ? vendor
              : "Unknown")
      << '\n'

      << "OpenGL Renderer: "
      << (renderer != nullptr
              ? renderer
              : "Unknown")
      << '\n'

      << "OpenGL Version: "
      << (gl_version != nullptr
              ? gl_version
              : "Unknown")
      << '\n'

      << "Model: "
      << options.model_path
      << '\n'

      << "Requested fur particles: "
      << options.particle_count
      << "\n\n";


  // OpenGL state

  glEnable(
      GL_DEPTH_TEST);

  glEnable(
      GL_BLEND);

  glBlendFunc(
      GL_SRC_ALPHA,
      GL_ONE_MINUS_SRC_ALPHA);


  // Load bunny

  Mesh bunny;


  if (!LoadPly(
          options.model_path,
          &bunny)) {

    std::cerr
        << "Failed to load the model.\n";

    glfwDestroyWindow(
        window);

    glfwTerminate();

    return 1;
  }


  std::cout

      << "Loaded mesh successfully.\n"

      << "Vertices:  "
      << bunny.vertices.size() / 6
      << '\n'

      << "Triangles: "
      << bunny.indices.size() / 3
      << "\n\n";


  // Bunny VAO

  GLuint bunny_vertex_array =
      0;

  GLuint bunny_vertex_buffer =
      0;

  GLuint bunny_index_buffer =
      0;


  glGenVertexArrays(
      1,
      &bunny_vertex_array);

  glGenBuffers(
      1,
      &bunny_vertex_buffer);

  glGenBuffers(
      1,
      &bunny_index_buffer);


  glBindVertexArray(
      bunny_vertex_array);


  glBindBuffer(
      GL_ARRAY_BUFFER,
      bunny_vertex_buffer);


  glBufferData(

      GL_ARRAY_BUFFER,

      static_cast<GLsizeiptr>(

          bunny.vertices.size() *
          sizeof(float)
      ),

      bunny.vertices.data(),

      GL_STATIC_DRAW
  );


  glBindBuffer(

      GL_ELEMENT_ARRAY_BUFFER,

      bunny_index_buffer
  );


  glBufferData(

      GL_ELEMENT_ARRAY_BUFFER,

      static_cast<GLsizeiptr>(

          bunny.indices.size() *
          sizeof(std::uint32_t)
      ),

      bunny.indices.data(),

      GL_STATIC_DRAW
  );


  constexpr GLsizei
      kVertexStride =
          6 * sizeof(float);


  /*
   * Position.
   */

  glVertexAttribPointer(

      0,

      3,

      GL_FLOAT,

      GL_FALSE,

      kVertexStride,

      nullptr
  );


  glEnableVertexAttribArray(
      0);


  /*
   * Normal.
   */

  glVertexAttribPointer(

      1,

      3,

      GL_FLOAT,

      GL_FALSE,

      kVertexStride,

      reinterpret_cast<void*>(
          3 * sizeof(float))
  );


  glEnableVertexAttribArray(
      1);


  glBindVertexArray(
      0);


  // Generate fur roots

  std::vector<float>
      fur_base_positions;


  std::vector<float>
      fur_normals;


  CreateFurData(

      bunny,

      options.particle_count,

      &fur_base_positions,

      &fur_normals
  );


  const std::size_t
      fur_particle_count =

          fur_base_positions.size() /
          3;


  std::cout

      << "Fur particles: "
      << fur_particle_count
      << '\n';


    // Allocate complete strands
    // ----------------------------------------------------------
    //
    // Each strand:
    //
    //     P0 P1 P2 P3 P4 P5 P6
    //
    // Each point:
    //
    //     X Y Z
    //
    // Therefore:
    // strands * 7 * 3 floats

  const std::size_t
      total_fur_points =

          fur_particle_count *

          static_cast<std::size_t>(
              kDisplayPointsPerStrand);


  const std::size_t
      total_fur_floats =

          total_fur_points *
          3;


  std::vector<float>
      fur_positions(

          total_fur_floats,

          0.0f
      );


  // Fur VAO/VBO

  GLuint fur_vertex_array =
      0;

  GLuint fur_vertex_buffer =
      0;


  glGenVertexArrays(
      1,
      &fur_vertex_array);


  glGenBuffers(
      1,
      &fur_vertex_buffer);


  glBindVertexArray(
      fur_vertex_array);


  glBindBuffer(
      GL_ARRAY_BUFFER,
      fur_vertex_buffer);


  glBufferData(

      GL_ARRAY_BUFFER,

      static_cast<GLsizeiptr>(

          fur_positions.size() *
          sizeof(float)
      ),

      fur_positions.data(),

      GL_DYNAMIC_DRAW
  );


  glVertexAttribPointer(

      0,

      3,

      GL_FLOAT,

      GL_FALSE,

      3 * sizeof(float),

      nullptr
  );


  glEnableVertexAttribArray(
      0);


  glBindVertexArray(
      0);


  // CUDA

  if (!InitializeParticleSimulation(

          fur_base_positions.data(),

          fur_normals.data(),

          fur_particle_count)) {

    std::cerr

        << "Failed to initialize CUDA "
        << "fur simulation.\n";


    glDeleteBuffers(
        1,
        &fur_vertex_buffer);

    glDeleteVertexArrays(
        1,
        &fur_vertex_array);

    glDeleteBuffers(
        1,
        &bunny_index_buffer);

    glDeleteBuffers(
        1,
        &bunny_vertex_buffer);

    glDeleteVertexArrays(
        1,
        &bunny_vertex_array);

    glfwDestroyWindow(
        window);

    glfwTerminate();

    return 1;
  }


  // Shaders

  const GLuint
      bunny_shader_program =

          CreateShaderProgram(

              "shaders/bunny.vert",

              "shaders/bunny.frag"
          );


  const GLuint
      fur_shader_program =

          CreateShaderProgram(

              "shaders/particle.vert",

              "shaders/particle.frag"
          );


  if (bunny_shader_program == 0 ||
      fur_shader_program == 0) {

    std::cerr

        << "Failed to create shader "
        << "programs.\n";


    ShutdownParticleSimulation();


    glDeleteProgram(
        bunny_shader_program);

    glDeleteProgram(
        fur_shader_program);

    glDeleteBuffers(
        1,
        &fur_vertex_buffer);

    glDeleteVertexArrays(
        1,
        &fur_vertex_array);

    glDeleteBuffers(
        1,
        &bunny_index_buffer);

    glDeleteBuffers(
        1,
        &bunny_vertex_buffer);

    glDeleteVertexArrays(
        1,
        &bunny_vertex_array);

    glfwDestroyWindow(
        window);

    glfwTerminate();

    return 1;
  }


  // Uniform locations

  const GLint
      bunny_mvp_location =

          glGetUniformLocation(

              bunny_shader_program,

              "uMvp"
          );


  const GLint
      fur_mvp_location =

          glGetUniformLocation(

              fur_shader_program,

              "uMvp"
          );


  // Model

  constexpr float
      kModelScale =
          2.0f;


  const glm::mat4 model =
      glm::scale(

          glm::mat4(1.0f),

          glm::vec3(
              kModelScale)
      );


  // Timing

  double previous_time =
      glfwGetTime();


  // MAIN LOOP

  while (

      glfwWindowShouldClose(
          window) ==

      GLFW_FALSE) {


    // Time

    const double current_time =
        glfwGetTime();


    float delta_time =
        static_cast<float>(

            current_time -
            previous_time
        );


    previous_time =
        current_time;


    if (delta_time < 0.0f) {

      delta_time =
          0.0f;
    }


    if (delta_time > 0.033f) {

      delta_time =
          0.033f;
    }


    // Camera controls

    const float
        camera_speed =
            0.75f *
            delta_time;


    /*
     * W = up.
     */

    if (glfwGetKey(
            window,
            GLFW_KEY_W) ==
        GLFW_PRESS) {

      camera.height +=
          camera_speed;
    }


    /*
     * S = down.
     */

    if (glfwGetKey(
            window,
            GLFW_KEY_S) ==
        GLFW_PRESS) {

      camera.height -=
          camera_speed;
    }


    /*
     * A = left.
     */

    if (glfwGetKey(
            window,
            GLFW_KEY_A) ==
        GLFW_PRESS) {

      camera.horizontal -=
          camera_speed;
    }


    /*
     * D = right.
     */

    if (glfwGetKey(
            window,
            GLFW_KEY_D) ==
        GLFW_PRESS) {

      camera.horizontal +=
          camera_speed;
    }


    /*
     * + = zoom in.
     */

    if (

        glfwGetKey(
            window,
            GLFW_KEY_EQUAL) ==
        GLFW_PRESS ||

        glfwGetKey(
            window,
            GLFW_KEY_KP_ADD) ==
        GLFW_PRESS) {

      camera.distance -=
          1.5f *
          delta_time;
    }


    /*
     * - = zoom out.
     */

    if (

        glfwGetKey(
            window,
            GLFW_KEY_MINUS) ==
        GLFW_PRESS ||

        glfwGetKey(
            window,
            GLFW_KEY_KP_SUBTRACT) ==
        GLFW_PRESS) {

      camera.distance +=
          1.5f *
          delta_time;
    }


    camera.distance =
        std::clamp(

            camera.distance,

            1.0f,

            8.0f
        );


    /*
     * R = reset.
     */

    if (glfwGetKey(
            window,
            GLFW_KEY_R) ==
        GLFW_PRESS) {

      camera.distance =
          3.0f;

      camera.height =
          0.20f;

      camera.horizontal =
          0.0f;

      camera.target_y =
          0.0f;
    }


    /*
     * ESC.
     */

    if (glfwGetKey(
            window,
            GLFW_KEY_ESCAPE) ==
        GLFW_PRESS) {

      glfwSetWindowShouldClose(

          window,

          GLFW_TRUE
      );
    }


    // CUDA UPDATE

    if (!UpdateParticleSimulation(
            delta_time)) {

      std::cerr
          << "CUDA fur update failed.\n";

      break;
    }


    // Retrieve all strand points

    if (!GetParticlePositions(

            fur_positions.data(),

            fur_particle_count)) {

      std::cerr

          << "Failed to retrieve fur "
          << "positions.\n";

      break;
    }


    // Upload fur

    glBindBuffer(

        GL_ARRAY_BUFFER,

        fur_vertex_buffer
    );


    glBufferSubData(

        GL_ARRAY_BUFFER,

        0,

        static_cast<GLsizeiptr>(

            fur_positions.size() *
            sizeof(float)
        ),

        fur_positions.data()
    );


    glBindBuffer(

        GL_ARRAY_BUFFER,

        0
    );


    // Viewport

    int width = 0;

    int height = 0;


    glfwGetFramebufferSize(

        window,

        &width,

        &height
    );


    if (height <= 0) {

      height =
          1;
    }


    glViewport(

        0,

        0,

        width,

        height
    );


    // Clear

    glClearColor(

        0.012f,
        0.012f,
        0.015f,
        1.0f
    );


    glClear(

        GL_COLOR_BUFFER_BIT |
        GL_DEPTH_BUFFER_BIT
    );


    // Camera matrices

    const float aspect =

        static_cast<float>(
            width) /

        static_cast<float>(
            height);


    const glm::vec3
        camera_position(

            camera.horizontal,

            camera.height,

            camera.distance
        );


    const glm::vec3
        camera_target(

            camera.horizontal,

            camera.target_y,

            0.0f
        );


    const glm::mat4 view =
        glm::lookAt(

            camera_position,

            camera_target,

            glm::vec3(

                0.0f,

                1.0f,

                0.0f
            )
        );


    const glm::mat4 projection =

        glm::perspective(

            glm::radians(
                50.0f),

            aspect,

            0.01f,

            100.0f
        );


    const glm::mat4 mvp =

        projection *
        view *
        model;


    // Draw bunny

    glUseProgram(

        bunny_shader_program
    );


    glUniformMatrix4fv(

        bunny_mvp_location,

        1,

        GL_FALSE,

        glm::value_ptr(
            mvp)
    );


    glBindVertexArray(

        bunny_vertex_array
    );


    glDrawElements(

        GL_TRIANGLES,

        static_cast<GLsizei>(

            bunny.indices.size()
        ),

        GL_UNSIGNED_INT,

        nullptr
    );


    /* Draw fur
     * --------------------------------------------------------
     *
     * Every strand is:
     *
     * P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6
     *
     * Therefore GL_LINE_STRIP is required.
     * 
     */

    glUseProgram(

        fur_shader_program
    );


    glUniformMatrix4fv(

        fur_mvp_location,

        1,

        GL_FALSE,

        glm::value_ptr(
            mvp)
    );


    glBindVertexArray(

        fur_vertex_array
    );


    /*
     * Draw every individual strand.
     */

    for (std::size_t strand = 0;
         strand <
             fur_particle_count;
         ++strand) {

      const GLint first =
          static_cast<GLint>(

              strand *

              static_cast<
                  std::size_t>(
                  kDisplayPointsPerStrand)
          );


      glDrawArrays(

          GL_LINE_STRIP,

          first,

          kDisplayPointsPerStrand
      );
    }


    // Display controls in title bar

    std::ostringstream title;


    title

        << "Dandelion Bunny"

        << " | W/S: Up/Down"

        << " | A/D: Left/Right"

        << " | +/-: Zoom"

        << " | Wheel: Zoom"

        << " | R: Reset"

        << " | ESC: Quit";


    glfwSetWindowTitle(

        window,

        title.str().c_str()
    );


    // Present

    glfwSwapBuffers(
        window);


    glfwPollEvents();
  }


  // CLEANUP

  ShutdownParticleSimulation();


  glDeleteProgram(

      fur_shader_program
  );


  glDeleteProgram(

      bunny_shader_program
  );


  glDeleteBuffers(

      1,

      &fur_vertex_buffer
  );


  glDeleteVertexArrays(

      1,

      &fur_vertex_array
  );


  glDeleteBuffers(

      1,

      &bunny_index_buffer
  );


  glDeleteBuffers(

      1,

      &bunny_vertex_buffer
  );


  glDeleteVertexArrays(

      1,

      &bunny_vertex_array
  );


  glfwDestroyWindow(
      window);


  glfwTerminate();


  return 0;
}