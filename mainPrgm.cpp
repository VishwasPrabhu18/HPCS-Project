#include<iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

//#define IMAGE_FILE "./Images/image100.jpg"
//#define PIXEL_FILE_RGB "./PixelResult/pixelValuesRGB.txt"
//#define PIXEL_FILE_GRAY_SCALE "./PixelResult/pixelValuesGrayScale.txt"

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

    /*for(int i=0; i<3; i++) {
        printf("%d %d %d\t\t%d %d %d\n", magnitudeArr[i][0],magnitudeArr[i][1],magnitudeArr[i][2], signArr[i][0], signArr[i][1], signArr[i][2]);
    }*/

    arr2[0] = boundaryTraversal_signedArr(signArr, 3, 3);
    boundaryTraversal_magnitudeArr(magnitudeArr, 3, 3, newArr);
    //arr2 = 4 different centroid values

    int a1Val=0, a2Val=0, a3Val=0;
    for(int i=0; i<8; i++) {
        int binArr[3] = {0, 0, 0};
        newArr[i] > 7 ? decToBin(7, binArr) : decToBin(newArr[i], binArr);
        //cout<<newArr[i] << " -> " << binArr[0] << " " <<binArr[1] << " " <<binArr[2]<<endl;
        a1Val = a1Val * 10 + binArr[0];
        a2Val = a2Val * 10 + binArr[1];
        a3Val = a3Val * 10 + binArr[2];
    }
    //cout<< a1Val << " " << a2Val << " " << a3Val<<endl;
    arr2[1] = binToDec1(a1Val);
    arr2[2] = binToDec1(a2Val);
    arr2[3] = binToDec1(a3Val);

}

void getCenterVal_3DLBP(int r, int c, int **arr, int **layer1, int **layer2, int **layer3, int **layer4) {
    //FILE *fp = fopen("./Layer_Values/layerValues.txt", "w");

    //fprintf(fp, "Layer1  Layer2  Layer3  Layer4\n");

    for(int i=0; i<r; i++) {
        for(int j=0; j<c; j++) {
            int a1[3][3], a2[4];
            if(i>0 && i<r-1 && j>0 && j<c-1) {
                int r1=0, c1=0;
                for(int k=i-1; k<i+2; k++) {
                    c1 = 0;
                    for(int l=j-1; l<j+2; l++) {
                        a1[r1][c1] = floor(arr[k][l]);
                        c1 += 1;
                    }
                    r1 += 1;
                }
                lbp_3d_calculation(a1, floor(arr[i][j]), a2);
                layer1[i][j] = a2[0];
                layer2[i][j] = a2[1];
                layer3[i][j] = a2[2];
                layer4[i][j] = a2[3];

                //fprintf(fp, "%d\t\t%d\t\t%d\t\t%d\n", a2[0], a2[1], a2[2], a2[3]);
            }
        }
    }

    //fclose(fp);
}

double convertToGrayscale(Vec3b &rgb) {
    double res = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
    return floor(res);
}

void method2(Mat_<Vec3b> img) {
    Mat_<Vec3b> img1=img, img2=img, img3=img, img4=img;

    int W = img.rows;
    int H = img.cols;

    int **layer1 = (int **)malloc(H * sizeof(int *));
    int **layer2 = (int **)malloc(H * sizeof(int *));
    int **layer3 = (int **)malloc(H * sizeof(int *));
    int **layer4 = (int **)malloc(H * sizeof(int *));
    int **arr = (int **)malloc(H * sizeof(int *));
    for (int r = 0; r < H; r++) {
        layer1[r] = (int *)malloc(W * sizeof(int));
        layer2[r] = (int *)malloc(W * sizeof(int));
        layer3[r] = (int *)malloc(W * sizeof(int));
        layer4[r] = (int *)malloc(W * sizeof(int));
        arr[r] = (int *)malloc(W * sizeof(int));
    }

    //FILE *fp1 = fopen(PIXEL_FILE_RGB, "w");
    //FILE *fp2 = fopen(PIXEL_FILE_GRAY_SCALE, "w");
    for(int i=0; i<H; i++) {
        for(int j=0; j<W; j++) {
            //fprintf(fp1, "[%d, %d, %d]\n", img(i, j)[0], img(i, j)[1], img(i, j)[2]);
            int res = convertToGrayscale(img(i, j));
            //fprintf(fp2, "%d\n", res);
            arr[i][j] = res;
        }
        //fprintf(fp2, "\n");
    }

    getCenterVal_3DLBP(H, W, arr, layer1, layer2, layer3, layer4);

    //displayImage(img1, layer1, "Layer 1");
    //displayImage(img2, layer2, "Layer 2");
    //displayImage(img3, layer3, "Layer 3");
    //isplayImage(img4, layer4, "Layer 4");
    
    //fclose(fp1);
    //fclose(fp2);
}

int main(int argc, char *argv[]) {
    auto startTime1 = high_resolution_clock::now();

    Mat_<Vec3b> image = imread(argv[1], IMREAD_COLOR);       //IMREAD_GRAYSCALE
    Mat imgGray;

    int imgW=image.cols, imgH=image.rows;
    //cout<<"( " << imgW << ", " << imgH << " )"<<endl;

    //imshow("Original Image", image);

    auto startTime = high_resolution_clock::now();

    method2(image);
    
    auto endTime = high_resolution_clock::now();

    cout<<"Total Time taken by 3D-LBP Method is : "<<duration_cast<milliseconds>(endTime - startTime).count()<<"ms"<<endl;
    cout<<"Total Time taken by the entire Program is : "<<duration_cast<milliseconds>(endTime - startTime1).count()<<"ms"<<endl;
    int k = waitKey(0);
    return 0;
}
