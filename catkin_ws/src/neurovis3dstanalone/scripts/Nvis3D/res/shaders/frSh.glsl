#version 330

// fragment-pixel output color
out vec4 outColor;

in vec2 texcoord;

uniform sampler2D p3d_Texture0;
uniform int arrSizeY;
uniform int arrSizeX;
uniform float myArray[1000];

void main() {



    int x = int(floor(arrSizeX*texcoord[0]));
    int y = int(floor(arrSizeY*texcoord[1]));


    float r = myArray[x*3 + y*3];
    float g = myArray[x*3 + y*3+1];
    float b = myArray[x*3 + y*3+2];

    outColor = vec4(r, g, b, 1.0);

}
