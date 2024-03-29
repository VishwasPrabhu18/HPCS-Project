 %%writefile main.cu

#include<iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include <math.h>
#include <stdlib.h>
#include <chrono>

#include <cuda_runtime.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define IMAGE_FILE "./image.jpg"

__device__ int binToDec1(int num) {
    int decimal_num = 0, base = 1, rem;

    while ( num > 0) {
        rem = num % 10;
        decimal_num = decimal_num + rem * base;
        num = num / 10;
        base = base * 2;
    }

    return decimal_num;
}

__device__ void decToBin(int num, int *arr) {
    int i=2;
    while(num > 0) {
        arr[i] = num % 2;
        num = num / 2;
        i--;
    }
}

__device__ int boundaryTraversal_signedArr(int arr[3][3], int N, int M) {
    int binVal = 0;
    for (int i = 0; i < M; i++) {
        binVal = binVal * 10 + arr[0][i];
    }

    for (int i = 1; i < N; i++) {
        binVal = binVal * 10 + arr[i][M-1];
    }

    if (N > 1) {
        for (int i = M - 2; i >= 0; i--) {
            binVal = binVal * 10 + arr[N - 1][i];
        }
    }

    if (M > 1) {
        for (int i = N - 2; i > 0; i--) {
            binVal = binVal * 10 + arr[i][0];
        }
    }

    return binToDec1(binVal);
}

__device__ void boundaryTraversal_magnitudeArr(int arr[3][3], int N, int M, int *newArr) {
    int pos=0;
    //converting 2D array to 1D array in clockwise manner
    for (int i = 0; i < M; i++) {
        newArr[pos] = arr[0][i];
        pos++;
    }
    for (int i = 1; i < N; i++) {
        newArr[pos] = arr[i][M-1];
        pos++;
    }
    if (N > 1) {
        for (int i = M - 2; i >= 0; i--) {
            newArr[pos] = arr[N - 1][i];
            pos++;
        }
    }
    if (M > 1) {
        for (int i = N - 2; i > 0; i--) {
            newArr[pos] = arr[i][0];
            pos++;
        }
    }
}

__device__ void lbp_3d_calculation(int arr1[3][3], int midVal, int *arr2) {
    int magnitudeArr[3][3], signArr[3][3], newArr[8];
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            if(i==1 && j==1) {
                magnitudeArr[i][j] = 0;
                signArr[i][j] = 0;
            } else {
                int val = arr1[i][j] - midVal;
                if(val > 0) {
                    magnitudeArr[i][j] = val;
                    signArr[i][j] = 1;
                } else {
                    magnitudeArr[i][j] = -val;
                    signArr[i][j] = 0;
                }
            }
        }
    }

    arr2[0] = boundaryTraversal_signedArr(signArr, 3, 3);
    boundaryTraversal_magnitudeArr(magnitudeArr, 3, 3, newArr);

    int a1Val=0, a2Val=0, a3Val=0;
    for(int i=0; i<8; i++) {
        int binArr[3] = {0, 0, 0};
        newArr[i] > 7 ? decToBin(7, binArr) : decToBin(newArr[i], binArr);

        a1Val = a1Val * 10 + binArr[0];
        a2Val = a2Val * 10 + binArr[1];
        a3Val = a3Val * 10 + binArr[2];
    }

    arr2[1] = binToDec1(a1Val);
    arr2[2] = binToDec1(a2Val);
    arr2[3] = binToDec1(a3Val);

}
 
__global__ void mainKernal(int *imgArr, int _rows, int _cols, int *layer1, int *layer2, int *layer3, int *layer4) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory declaration
    __shared__ int sharedImg[18][18];

    // Copy data to shared memory
    if(threadIdx.y < 18 && threadIdx.x < 18 && row < _rows && col < _cols) {
        sharedImg[threadIdx.y][threadIdx.x] = imgArr[row * _cols + col];
    }
    __syncthreads();

    // Indices for accessing shared memory
    int sharedRow = threadIdx.y + 1;
    int sharedCol = threadIdx.x + 1;

    if(row < _rows-1 && col < _cols-1) {
        int a1[3][3], a2[4];
        // Fetch data from shared memory
        for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
                a1[i][j] = sharedImg[sharedRow + i - 1][sharedCol + j - 1];
            }
        }

        int midVal = a1[1][1];

        lbp_3d_calculation(a1, midVal, a2);

        // Same calculations as before, modified for shared memory access
        int magnitudeArr[3][3], signArr[3][3], newArr[8];
        for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
                if(i==1 && j==1) {
                    magnitudeArr[i][j] = 0;
                    signArr[i][j] = 0;
                } else {
                    int val = a1[i][j] - midVal;
                    if(val > 0) {
                        magnitudeArr[i][j] = val;
                        signArr[i][j] = 1;
                    } else {
                        magnitudeArr[i][j] = -val;
                        signArr[i][j] = 0;
                    }
                }
            }
        }

        //layer1[idx] = boundaryTraversal_signedArr(signArr, 3, 3);

        boundaryTraversal_magnitudeArr(magnitudeArr, 3, 3, newArr);

        int a1Val=0, a2Val=0, a3Val=0;
        for(int i=0; i<8; i++) {
            int binArr[3] = {0, 0, 0};
            newArr[i] > 7 ? decToBin(7, binArr) : decToBin(newArr[i], binArr);

            a1Val = a1Val * 10 + binArr[0];
            a2Val = a2Val * 10 + binArr[1];
            a3Val = a3Val * 10 + binArr[2];
        }

        /*layer2[idx] = binToDec1(a1Val);
        layer3[idx] = binToDec1(a2Val);
        layer4[idx] = binToDec1(a3Val);*/

        layer1[row * _cols + col] = boundaryTraversal_signedArr(signArr, 3, 3);
        layer2[row * _cols + col] = binToDec1(a1Val);
        layer3[row * _cols + col] = binToDec1(a2Val);
        layer4[row * _cols + col] = binToDec1(a3Val);
    }
}


int main(int argc, char *argv[]) {

    auto startTime = high_resolution_clock::now();

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    Mat image = imread(IMAGE_FILE, IMREAD_GRAYSCALE);
    int W = image.cols, H = image.rows;
    int size = W * H * sizeof(int);

    int imgArr[H * W], layer1[H * W], layer2[H * W], layer3[H * W], layer4[H * W];
    for(int i=0; i<H; i++) {
        for(int j=0; j<W; j++) {
            imgArr[i * W + j] = (int)image.at<uchar>(i, j);
        }
    }


    Mat result_image_layer1(H, W, CV_8UC1);
    Mat result_image_layer2(H, W, CV_8UC1);
    Mat result_image_layer3(H, W, CV_8UC1);
    Mat result_image_layer4(H, W, CV_8UC1);

    // Creating the 4 Layers
    int *d_imgArr, *d_layer1, *d_layer2, *d_layer3, *d_layer4;

    cudaMalloc((void**)&d_imgArr, size);
    cudaMalloc((void**)&d_layer1, size);
    cudaMalloc((void**)&d_layer2, size);
    cudaMalloc((void**)&d_layer3, size);
    cudaMalloc((void**)&d_layer4, size);

    cudaMemcpy(d_imgArr, imgArr, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y);

    cudaEventRecord(start, 0);

    mainKernal<<<gridDim, blockDim>>>(d_imgArr, H, W, d_layer1, d_layer2, d_layer3, d_layer4);

    cudaEventRecord(end, 0);

    cudaMemcpy(layer1, d_layer1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(layer2, d_layer2, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(layer3, d_layer3, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(layer4, d_layer4, size, cudaMemcpyDeviceToHost);

    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, end);
    printf("Kernal Execution Time : %.10fms\n", elapsed_time_ms);

    for(int i=0; i<H; i++) {
      for(int j=0; j<W; j++) {
        result_image_layer1.at<uchar>(i, j) = static_cast<uchar>(layer1[i * W + j]);
        result_image_layer2.at<uchar>(i, j) = static_cast<uchar>(layer2[i * W + j]);
        result_image_layer3.at<uchar>(i, j) = static_cast<uchar>(layer3[i * W + j]);
        result_image_layer4.at<uchar>(i, j) = static_cast<uchar>(layer4[i * W + j]);
      }
    }

    imwrite("./output/Layer1-Img.jpg", result_image_layer1);
    imwrite("./output/Layer2-Img.jpg", result_image_layer2);
    imwrite("./output/Layer3-Img.jpg", result_image_layer3);
    imwrite("./output/Layer4-Img.jpg", result_image_layer4);

    cudaFree(d_imgArr);
    cudaFree(d_layer1);
    cudaFree(d_layer2);
    cudaFree(d_layer3);
    cudaFree(d_layer4);

    auto endTime = high_resolution_clock::now();

    cout<<"\nTime Taken by entire 3D-LBP program is : "<<duration_cast<milliseconds>(endTime - startTime).count()<<"ms"<<endl;
    return 0;
}