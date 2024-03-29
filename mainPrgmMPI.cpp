#include<iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#include <mpi.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

#define IMAGE_FILE "./Images/image700.jpg"
//#define IMAGE_FILE "./Data/image2.jpg"
//#define PIXEL_FILE_RGB "./results/PixelResult/pixelValuesRGB.txt"
//#define PIXEL_FILE_GRAY_SCALE "./results/PixelResult/pixelValuesGrayScale.txt"


int binToDec1(int num) {
    int decimal_num = 0, base = 1, rem;

    while ( num > 0) {
        rem = num % 10;
        decimal_num = decimal_num + rem * base;
        num = num / 10;
        base = base * 2;
    }

    return decimal_num;
}

void decToBin(int num, int *arr) {
    int i=2;
    while(num > 0) {
        arr[i] = num % 2;
        num = num / 2;
        i--;
    }
}

int boundaryTraversal_signedArr(int arr[3][3], int N, int M) {
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

void boundaryTraversal_magnitudeArr(int arr[3][3], int N, int M, int *newArr) {
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

void lbp_3d_calculation(int arr1[3][3], int midVal, int *arr2) {
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

int main(int argc, char *argv[]) {
    auto startTime1 = high_resolution_clock::now();

    int rank, size;

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    
    int W = image.cols, H = image.rows;
    Mat result_image_layer1(image.rows, image.cols, CV_8UC1);
    Mat result_image_layer2(image.rows, image.cols, CV_8UC1); 
    Mat result_image_layer3(image.rows, image.cols, CV_8UC1); 
    Mat result_image_layer4(image.rows, image.cols, CV_8UC1); 

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Time Calculation Begin
    
    auto startTime = high_resolution_clock::now();
    
    int row_per_process = H / size;
    int start_row1 = rank*row_per_process;
    int start_row2 = (rank == 0) ? 0 : start_row1 - 1;
    int end_row = (rank==size-1) ? H : start_row1+row_per_process;
    int totalRows = end_row - start_row2;

    // Creating the 4 Layers
    int layer1[H][W], layer2[H][W], layer3[H][W], layer4[H][W], arr[H][W];

    for(int i=start_row2; i<end_row; i++) {
        for(int j=0; j<W; j++) {
            arr[i][j] = (int)image.at<uchar>(i, j);
        }
    }

    int r = totalRows, c = W;
    for(int i=start_row2, x=0; i<end_row; i++, x++) {
        for(int j=0; j<W; j++) {
            int a1[3][3], a2[4];
            if(x>0 && x<r-1 && j>0 && j<c-1) {
                int r1=0, c1=0;
                for(int k=i-1; k<i+2; k++) {
                    c1 = 0;
                    for(int l=j-1; l<j+2; l++) {
                        a1[r1][c1] = floor(arr[k][l]);
                        c1 += 1;
                    }
                    r1 += 1;
                }
                //int midVal = floor(arr[i][j]);
                //lbp_3d_calculation(a1, floor(arr[i][j]), a2);
                int midVal = a1[1][1];
                lbp_3d_calculation(a1, midVal, a2);
                int magnitudeArr[3][3], signArr[3][3], newArr[8], arr2[4];
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
                
                layer1[i][j] = boundaryTraversal_signedArr(signArr, 3, 3);

                boundaryTraversal_magnitudeArr(magnitudeArr, 3, 3, newArr);

                int a1Val=0, a2Val=0, a3Val=0;
                for(int i=0; i<8; i++) {
                    int binArr[3] = {0, 0, 0};
                    newArr[i] > 7 ? decToBin(7, binArr) : decToBin(newArr[i], binArr);
                    
                    a1Val = a1Val * 10 + binArr[0];
                    a2Val = a2Val * 10 + binArr[1];
                    a3Val = a3Val * 10 + binArr[2];
                }
                
                layer2[i][j] = binToDec1(a1Val);
                layer3[i][j] = binToDec1(a2Val);
                layer4[i][j] = binToDec1(a3Val);

                result_image_layer1.at<uchar>(i, j) = static_cast<uchar>(layer1[i][j]);
                result_image_layer2.at<uchar>(i, j) = static_cast<uchar>(layer2[i][j]);
                result_image_layer3.at<uchar>(i, j) = static_cast<uchar>(layer3[i][j]);
                result_image_layer4.at<uchar>(i, j) = static_cast<uchar>(layer4[i][j]);
            }
        }
    }

    // Time Calculation Ends
    auto endTime = high_resolution_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(result_image_layer1.data + start_row2 * result_image_layer1.cols, row_per_process * result_image_layer1.cols,
           MPI_CHAR, result_image_layer1.data, row_per_process * result_image_layer1.cols, MPI_CHAR,
           0, MPI_COMM_WORLD);
    MPI_Gather(result_image_layer2.data + start_row2 * result_image_layer2.cols, row_per_process * result_image_layer2.cols,
           MPI_CHAR, result_image_layer2.data, row_per_process * result_image_layer2.cols, MPI_CHAR,
           0, MPI_COMM_WORLD);
    MPI_Gather(result_image_layer3.data + start_row2 * result_image_layer3.cols, row_per_process * result_image_layer3.cols,
           MPI_CHAR, result_image_layer3.data, row_per_process * result_image_layer3.cols, MPI_CHAR,
           0, MPI_COMM_WORLD);
    MPI_Gather(result_image_layer4.data + start_row2 * result_image_layer4.cols, row_per_process * result_image_layer4.cols,
           MPI_CHAR, result_image_layer4.data, row_per_process * result_image_layer4.cols, MPI_CHAR,
           0, MPI_COMM_WORLD);

     auto endTime1 = high_resolution_clock::now();

    if(rank == 0) {
        /*imwrite("./output/Layer1-Img.jpg", result_image_layer1);
        imwrite("./output/Layer2-Img.jpg", result_image_layer2);
        imwrite("./output/Layer3-Img.jpg", result_image_layer3);
        imwrite("./output/Layer4-Img.jpg", result_image_layer4);

        Mat layer1_img = imread("./output/Layer1-Img.jpg", IMREAD_GRAYSCALE);
        Mat layer2_img = imread("./output/Layer2-Img.jpg", IMREAD_GRAYSCALE);
        Mat layer3_img = imread("./output/Layer3-Img.jpg", IMREAD_GRAYSCALE);
        Mat layer4_img = imread("./output/Layer4-Img.jpg", IMREAD_GRAYSCALE);

        imshow("Original Image", image);
        imshow("Layer1 Image", layer1_img);
        imshow("Layer2 Image", layer2_img);
        imshow("Layer3 Image", layer3_img);
        imshow("Layer4 Image", layer4_img); */

        cout<<"\nTime Taken for the 3D-LBP Calculation is : "<<duration_cast<milliseconds>(endTime - startTime).count()<<"ms"<<endl;
        cout<<"Time Taken by the  entire 3D-LBP program is : "<<duration_cast<milliseconds>(endTime1 - startTime1).count()<<"ms"<<endl;  
    }


    waitKey(0);
    MPI_Finalize();

    return 0;
}
