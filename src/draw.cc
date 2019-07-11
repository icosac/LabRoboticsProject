#include "draw.hh"

int DW::init(	uint x, 
							uint y, 
							GLfloat vertices_buffer){
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
	window = glfwCreateWindow( x, y, "Map", NULL, NULL);
	if( window == NULL ){
	    fprintf( stderr, "Failed to open GLFW window.");
	    glfwTerminate();
	    return -1;
	}
	glfwMakeContextCurrent(window); // Initialize GLEW
	glewExperimental=true; // Needed in core profile
	if (glewInit() != GLEW_OK) {
	    fprintf(stderr, "Failed to initialize GLEW\n");
	    return -1;
	}

	//I don't know why this is needed, but it is
	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// Generate 1 buffer, put the resulting identifier inside map_buffer
	glGenBuffers(1, &map_buffer);
	// The following commands will talk about our 'triangle1_buff' buffer
	glBindBuffer(GL_ARRAY_BUFFER, map_buffer);
	// Give our vertices to OpenGL.
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_buffer), vertices_buffer, GL_DYNAMIC_DRAW);

	return 0;
}

void DW::changeBuffer (GLfloat* vertices_buffer, uint dim){
	glBufferData(GL_ARRAY_BUFFER, dim, vertices_buffer, GL_DYNAMIC_DRAW);
}