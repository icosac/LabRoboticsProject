#include<cstdlib>
#include<cstdio>
#include<iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <OpenGL/gl.h>
#include <OpenGl/glu.h>
#include <GLUT/glut.h>

#include "shader.hh"
#include <unistd.h>
#include <thread>

using namespace glm;

GLFWwindow* window;

void fun1 (GLuint triangle_buff, GLuint programID){
	std::cout << "Ok" << std::endl;	
	int i=0;
	do{
    // Clear the screen. It's not mentioned before Tutorial 02, but it can cause flickering, so it's there nonetheless.
    glClear( GL_COLOR_BUFFER_BIT );
    glUseProgram(programID);
		// 1st attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, triangle_buff);

		glVertexAttribPointer(
			0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
			2,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);
		// Draw the triangle !
		glDrawArrays(GL_TRIANGLES, 0, 6); // Starting from vertex 0; 3 vertices total -> 1 triangle
		glDisableVertexAttribArray(0);
		// Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();
 
    sleep(3);
    GLfloat triangle1[] = {
			   -1.0f, 0.0f,
			   -0.5f, 1.0f,
			   0.0f, 0.0f,
			   1.0f, 0.0f,
			   0.5f, 1.0f,
			   0.0f, 0.0f,
		};
		glBufferData(GL_ARRAY_BUFFER, sizeof(triangle1), triangle1, GL_DYNAMIC_DRAW);
	// Check if the ESC key was pressed or the window was closed
	} while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
	       glfwWindowShouldClose(window) == 0 );
}

int main(){
	// Initialise GLFW
	glewExperimental = true; // Needed for core profile
	if( !glfwInit() )
	{
	    fprintf( stderr, "Failed to initialize GLFW\n" );
	    return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 

	// Open a window and create its OpenGL context
	window = glfwCreateWindow( 8000, 4000, "Tutorial 01", NULL, NULL);
	if( window == NULL ){
	    fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
	    glfwTerminate();
	    return -1;
	}
	glfwMakeContextCurrent(window); // Initialize GLEW
	glewExperimental=true; // Needed in core profile
	if (glewInit() != GLEW_OK) {
	    fprintf(stderr, "Failed to initialize GLEW\n");
	    return -1;
	}

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	GLfloat triangle2[] = {
			0.0f
	};

	// GLfloat g_vertex_buffer_data[6]; 
	// g_vertex_buffer_data[0]=-1.0f;
	// g_vertex_buffer_data[1]=-1.0f;
	// g_vertex_buffer_data[2]=1.0f;
	// g_vertex_buffer_data[3]=-1.0f;
	// g_vertex_buffer_data[4]=0.0f;
	// g_vertex_buffer_data[5]=1.0f;

	// This will identify our vertex buffer
	GLuint triangle_buff;
	// Generate 1 buffer, put the resulting identifier in triangle2_buff
	glGenBuffers(1, &triangle_buff);
	// The following commands will talk about our 'triangle_buff' buffer
	glBindBuffer(GL_ARRAY_BUFFER, triangle_buff);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, sizeof(triangle2), triangle2, GL_DYNAMIC_DRAW);

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	// Create and compile our GLSL program from the shaders
	GLuint programID = LoadShaders( "test/SimpleVertexShader.vertexshader", "test/SimpleFragmentShader.fragmentshader" );
	
	pid_t pid = fork();
  if (pid == 0){
    // child process
    fun1(triangle_buff, programID);
  }
  else if (pid > 0)
  {
    std::vector<int> v;
		for (int g=0; g<10; g++){
			int a=0;
			std::cin >> a;
			v.push_back(a);
		}
		for (int el : v){
			std::cout << el << std::endl;
		}
  }
  else
  {
    // fork failed
    printf("fork() failed!\n");
    return 1;
  }

	return 0;

}
