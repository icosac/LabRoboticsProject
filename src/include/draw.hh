#ifndef DRAW_HH
#define DRAW_HH

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <OpenGL/gl.h>
#include <OpenGl/glu.h>
#include <GLUT/glut.h>

using namespace glm;

typedef uint unsigned int;

namespace DW {
	GLFWwindow* window;
	GLuint map_buffer;

	void init (x, y, GLfloat* vertices_buffer={0.0f});
	void changeBuffer (GLfloat* vertices_buffer, uint dim);
}

#endif