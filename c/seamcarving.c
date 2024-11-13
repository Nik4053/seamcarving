#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static void die(const char* msg){
    perror(msg);
    exit(EXIT_FAILURE);
}
// alternative implementation of rgb2gray
// static void rgb2gray_alt(const unsigned char *img,unsigned char *result, size_t width, size_t height){
//     float gamma = 2.2f;
    
//     #pragma omp parallel for
//     for(size_t y = 0; y < height;y++){
//         size_t pos = y * width*3;
//         for(size_t x=0;x<width*3;x+=3){
//             pos +=3; 
//             float R = img[pos+0];
//             float G = img[pos+1];
//             float B = img[pos+2];
//             float Clin = 0.2126f * powf(R,gamma) + 0.7152f * powf(G,gamma) + 0.0722f * powf(B,gamma);
//             float Csrgb = 0;
//             if(Clin <= 0.0031308f){
//                 Csrgb = 12.92f * Clin;
//             }else{
//                 Csrgb = 1.055f * powf(Clin,1/2.4f)-0.055f;
//             }
//             result[y*width+x/3] =(char)Csrgb;
//         }
//     }
// }


static void rgb2gray(const unsigned char *img,unsigned char *result, size_t width, size_t height){
    #pragma omp parallel for
    for(size_t y = 0; y < height;y++){
        size_t pos = y * width*3;
        for(size_t x=0;x<width*3;x+=3){
            pos +=3; 
            float R = img[pos+0];
            float G = img[pos+1];
            float B = img[pos+2];
            float grey = 0.2989f * R + 0.5870f * G + 0.1140 * B;
            result[y*width+x/3] =(char)grey;
        }
    }
}

/*
    @param img: input img of size width*height
    @param result: resulting img of size width*height
    @param scale: scale factor to apply to the result
    @note: the result of applying kernel to img will be added to the result array. 
    If you want to overwrite the result array, you have to set it to 0 before calling this function

*/
static void applyKernel(const unsigned char *img,unsigned char *result, size_t width, size_t height, float kernel[3][3], float scale){
    
#pragma omp parallel for collapse(2)
    for(size_t y = 0; y < height;y++){
        for(size_t x=0; x < width ;x++){
            float sum = 0;
            // 3x3 kernel
            for(int j=0;j<3;j++){
                for(int i=0;i<3;i++){
                    if(x+i==0 || x+i>=width+1 || y+j==0 || y+j>=height+1) continue;
                    sum += img[(y+j-1)*width+(x+i-1)] * kernel[j][i];
                }
            }
            sum = abs(sum);
            sum = ( sum + result[y*width+x] ) * scale;
            result[y*width+x] = (char)sum;
        }
    }
}

static void getEnergyMap(const unsigned char *img,float *result, size_t width, size_t height){
    // last row
    for(size_t x=0; x < width ;x++){
        result[(height-1)*width+x] = img[(height-1)*width+x];
    }
    for (int y = height-2; y >= 0; y--){
#pragma omp parallel for
        for(size_t x=0; x < width ;x++){
            int best = result[(y+1)*width+x];
            if(x>0 && result[(y+1)*width+x-1] < best) best = result[(y+1)*width+x-1];
            if(x<width-1 && result[(y+1)*width+x+1] < best) best = result[(y+1)*width+x+1];
            result[y*width+x] = img[y*width+x] + best;
        }
    }
}

static void scaleImg(const float *img,unsigned char *result, size_t width, size_t height, float scale){
    for(size_t y = 0; y < height;y++){
        for(size_t x=0; x < width ;x++){
            result[y*width+x] = (char)(img[y*width+x] * scale);
        }
    }
}


static void greedyPath(const float *img,unsigned int *resultingPath, size_t width, size_t height){
    // starting point
    float best = img[0];
    int bestX = 0;
    for(size_t x=0; x < width ;x++){
        if(img[x] < best){
            best = img[x];
            bestX = x;
        }
    }
    resultingPath[0] = bestX;
    // rest of the path
    for(size_t y = 1; y < height;y++){
        int best = img[y*width+resultingPath[y-1]];
        int bestX = resultingPath[y-1];
        if(resultingPath[y-1]>0 && img[y*width+resultingPath[y-1]-1] < best){
            best = img[y*width+resultingPath[y-1]-1];
            bestX = resultingPath[y-1]-1;
        }
        if(resultingPath[y-1]<width-1 && img[y*width+resultingPath[y-1]+1] < best){
            best = img[y*width+resultingPath[y-1]+1];
            bestX = resultingPath[y-1]+1;
        }
        resultingPath[y] = bestX;
    }
}

/*
    @param img: input img of size width*height
    @param result: resulting img of size (width-1)*height
    @param seam: seam to remove
    @note: result and img can be the same array
*/
static void removeSeam(const unsigned char *img,unsigned char *result, size_t width, size_t height, unsigned int *seam){
    for(size_t y = 0; y < height;y++){
        memcpy(result+y*(width-1),img+y*width,seam[y]);
        memcpy(result+y*(width-1)+seam[y],img+y*width+seam[y]+1,width-seam[y]-1);
    }
}
/*
    @param img: input img of size width*height*3
    @param result: resulting img of size (width-1)*height*3
    @param seam: seam to remove
    @note: result and img can be the same array
*/
static void removeSeamRGB(const unsigned char *img,unsigned char *result, size_t width, size_t height, unsigned int *seam){
    for(size_t y = 0; y < height;y++){
        memcpy(result+y*(width-1)*3,img+y*width*3,seam[y]*3);
        memcpy(result+y*(width-1)*3+seam[y]*3,img+y*width*3+(seam[y]+1)*3,(width-seam[y]-1)*3);
    }
}

/*
    Fills the seam in the image with white pixels.
*/
static void fillSeam(unsigned char *img, size_t width, size_t height, unsigned int *seam){
    for(size_t y = 0; y < height;y++){
        for(size_t x=0; x < width ;x++){
            if(x==seam[y]) {
                img[y*width+x] = 255;
                break;
            }
        }
    }
}


static void clearArray(unsigned char *array, size_t width, size_t height){
    for(size_t y = 0; y < height;y++){
        for(size_t x=0; x < width ;x++){
            array[y*width+x] = 0;
        }
    }
}

int main(int argc, char **argv)
{
    int greedy = 0; // greedy version does not recalculate the kernel image. May lead to artifacts
    printf("Running using %d threads!\n", omp_get_max_threads());
    if (argc != 3) {
        printf("USAGE: {targetIMG} {numberOfIter}\n");
        exit(EXIT_SUCCESS);
    }


    int img_w,img_h,img_c; // image width,image height, number of original channels per pixel
    unsigned char *img;
    printf("Loading Image: %s\n", argv[1]);
    img = stbi_load(argv[1], &img_w, &img_h, &img_c, 3); // we only need RGB not RGBA so we choose level 3
    if(img == NULL) die("ERROR loading Image");
    if(img_c!=3) printf("WARNING UNSUPPORTED OPERATION: Image has %d channels per pixel. Output will contain %d channels per pixel\n", img_c , 3);
    
    if (img_w < 3 || img_h < 3) die("Image is too small");
    
    unsigned char *greyImg = (unsigned char*) calloc(img_w*img_h,sizeof(unsigned char));
    if(greyImg==NULL) die("Could not alloc greyImg");
    unsigned char *kernelImg = (unsigned char*) calloc(img_w*img_h,sizeof(unsigned char));
    if(greyImg==NULL) die("Could not alloc kernelImg");

    // convert to grey scale
    rgb2gray(img,greyImg,img_w,img_h);


    float *energyMap = (float*) calloc((img_w)*(img_h),sizeof(*energyMap));
    if(energyMap==NULL) die("Could not alloc energyMap");
    
    float sobelX[3][3] = {{ -0.125,0.0,0.125},{-0.25,0.0,0.25},{-0.125,0.0,0.125}};
    float sobelY[3][3] = {{ -0.125,-0.25,-0.125},{0.0,0.0,0.0},{0.125,0.25,0.125}};
    
    clearArray(kernelImg,img_w,img_h);
    applyKernel(greyImg,kernelImg,img_w,img_h,sobelX,1);
    applyKernel(greyImg,kernelImg,img_w,img_h,sobelY,0.5f);
    
    unsigned int path[img_h];
    int iter = atoi(argv[2]);
    if (iter > img_w-1) die("Number of iterations is too large. It cannot be larger than the width of the image minus 1");
    
    for(int i=0;i<iter;i++){
        if (i%50==0)
            printf("Iteration: %d\n",i);
        if (!greedy){
            applyKernel(greyImg,kernelImg,img_w,img_h,sobelX,1);
            applyKernel(greyImg,kernelImg,img_w,img_h,sobelY,0.5f);
        }

        getEnergyMap(kernelImg,energyMap,img_w,img_h);
        greedyPath(energyMap,path,img_w,img_h);
        
        if (greedy){
            removeSeam(kernelImg,kernelImg,img_w,img_h,path);
        }else{
            removeSeam(greyImg,greyImg,img_w,img_h,path);
        }
        removeSeamRGB(img,img,img_w,img_h,path);
        img_w--;
    }
    
    rgb2gray(img,greyImg,img_w,img_h);
    // fillSeam(greyImg,img_w,img_h,path);
    stbi_write_jpg("out_rgb.jpg",img_w,img_h, 3,img,100 );

    stbi_write_jpg("out.jpg",img_w,img_h, 1,greyImg,100 );

    free(greyImg);
    free(kernelImg);
    free(energyMap);
    stbi_image_free(img);
    
}