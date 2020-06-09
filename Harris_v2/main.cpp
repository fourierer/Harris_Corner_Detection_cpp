//
//  main.cpp
//  Harris
//
//  Created by Zheng Sun on 2020/6/3.
//  Copyright © 2020 Zheng Sun. All rights reserved.
//

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;
/*
RGB转换成灰度图像的一个常用公式是：
Gray = R*0.299 + G*0.587 + B*0.114
*/
//******************灰度转换函数*************************
//第一个参数image输入的彩色RGB图像的引用，第二个参数imageGray是转换后输出的灰度图像的引用；
void ConvertRGB2GRAY(const Mat &image, Mat &imageGray);
 
//******************Sobel卷积因子计算X、Y方向梯度和梯度方向角********************
//第一个参数imageSourc原始灰度图像；
//第二个参数imageSobelX是X方向梯度图像；
//第三个参数imageSobelY是Y方向梯度图像；
//第四个参数pointDrection是梯度方向角数组指针
void SobelGradDirction(Mat &imageSource, Mat &imageSobelX, Mat &imageSobelY);
 
//******************计算Sobel的X方向梯度幅值的平方*************************
//第一个参数imageGradX是X方向梯度图像；
//第二个参数SobelAmpXX是输出的X方向梯度图像的平方；
void SobelXX(const Mat imageGradX, Mat_<float> &SobelAmpXX);
 
//******************计算Sobel的Y方向梯度幅值的平方*************************
//第一个参数imageGradY是Y方向梯度图像；
//第二个参数SobelAmpXX是输出的Y方向梯度图像的平方；
void SobelYY(const Mat imageGradY, Mat_<float> &SobelAmpYY);
 
//******************计算Sobel的XY方向梯度幅值的乘积*************************
//第一个参数imageGradX是X方向梯度图像；
//第二个参数imageGradY是Y方向梯度图像；
//第三个参数SobelAmpXY是输出的XY方向梯度图像；
void SobelXY(const Mat imageGradX, const Mat imageGradY, Mat_<float> &SobelAmpXY);
 
//****************计算二维高斯核*****************
//第一个参数size是代表的卷积核的边长的大小；
//第二个参数sigma表示的是sigma的大小；
double **getTwoGaussionArray(int size, double sigma);

//****计算局部特征结果矩阵M的特征值和响应函数R = (A*B - C^2) - k*(A+B)^2******
//假设M矩阵的4个元素分别为A，C，C，B，则有：
//Trace(M)=lambda_1+lambda_2=A+B
//Det(M)=lambda_1*lambda_2=A*B-C^2
//计算输出响应函数的矩阵
//****************************************************************************
void harrisResponse(Mat_<float> &imageSobelXX, Mat_<float> &imageSobelYY, Mat_<float> &imageSobelXY, Mat_<float> &resultData,double **arr, float k, int rsize);
 
 
//***********非极大值抑制和满足阈值及某邻域内的局部极大值为角点**************
//第一个参数是响应函数的矩阵
//第二个参数是输入的灰度图像
//第三个参数表示的是输出的角点检测到的结果图
void LocalMaxValue(Mat_<float> &resultData, Mat &srcGray, Mat &ResultImage,int rsize, int threshold);


int main()
{
    //const Mat srcImage = imread("/Users/momo/Desktop/iron_man.jpg");
    const Mat srcImage = imread("iron_man.jpg");
    //const Mat srcImage = imread("square.jpg");
    cout<<srcImage.size()<<endl;
    if (!srcImage.data)
    {
        printf("could not load image...\n");
        return -1;
    }
    
    Mat srcGray;
    ConvertRGB2GRAY(srcImage, srcGray);
    Mat imageSobelX;
    Mat imageSobelY;
    Mat resultImage;
    Mat_<float> imageSobelXX;
    Mat_<float> imageSobelYY;
    Mat_<float> imageSobelXY;
    Mat_<float> HarrisRespond;//响应函数矩阵的值，根据响应函数来判断该点是否为角点
    
    //计算Soble的XY梯度
    SobelGradDirction(srcGray, imageSobelX, imageSobelY);
    //cout<<"imageSobelX:"<<imageSobelX.size()<<endl;
    //cout<<"imageSobelY:"<<imageSobelY.size()<<endl;
    //计算X,Y方向的梯度的平方
    SobelXX(imageSobelX, imageSobelXX);//(Ix)^2
    SobelYY(imageSobelY, imageSobelYY);//(Iy)^2
    SobelXY(imageSobelX, imageSobelY, imageSobelXY);//(Ix)(Iy)
    //cout<<"imageSobelXX:"<<imageSobelXX.size()<<endl;
    //cout<<"imageSobelYY:"<<imageSobelYY.size()<<endl;
    //cout<<"imageSobelXY:"<<imageSobelXY.size()<<endl;
    
    int size = 5;//高斯核尺寸，size*size
    int rsize = size / 2;//高斯核半径
    double **gaussian = getTwoGaussionArray(size,1);//尺寸为3的二维高斯卷积核
    /*
    //测试二维高斯卷积核
    for (int i = 0; i<size; i++)
    {
        for (int j = 0; j<size; j++)
            cout<<gause[i][j]<<" ";
        cout<<endl;
    }
    */
    float k = 0.05;//k是响应函数中的经验参数，一般在0.04-0.06之间
    int threshold = 50000000;//响应函数的阈值,square.jpg阈值为100000时效果较好，iron_man.jpg阈值为50000000时效果较好
    harrisResponse(imageSobelXX, imageSobelYY, imageSobelXY, HarrisRespond, gaussian, k, rsize);
    LocalMaxValue(HarrisRespond, srcGray, resultImage, rsize, threshold);
    imshow("srcGray", srcGray);
    //imshow("imageSobelX", imageSobelX);//SobelX检测竖直方向的边缘
    //imshow("imageSobelY", imageSobelY);//SobelY检测水平方向的边缘
    imshow("resultImage", resultImage);
    waitKey(0);
    return 0;
}

//RGB图转灰度图
void ConvertRGB2GRAY(const Mat &image, Mat &imageGray)
{
    if (!image.data || image.channels() != 3)
    {
        return;
    }
    //创建一张单通道的灰度图像
    /*
    一般的图像文件格式使用的是Unsigned 8bits，CvMat矩阵对应的参数类型就是CV_8UC1，CV_8UC2，CV_8UC3
     （最后的1、2、3表示通道数，譬如RGB3通道就用CV_8UC3）
     而float 是32位的，对应CvMat数据结构参数为：CV_32FC1，CV_32FC2，CV_32FC3
     double是64bits，对应CvMat数据结构参数：CV_64FC1，CV_64FC2，CV_64FC3等
    */
    imageGray = Mat::zeros(image.size(), CV_8UC1);//Mat是头文件里的类，用类Mat中的函数zeros
    //cout<<image.size()<<endl;//(546,300)
    //cout<<imageGray.rows<<endl;//300
    //取出存储图像像素的数组的指针
    uchar *pointImage = image.data;
    uchar *pointImageGray = imageGray.data;
    //取出图像每行所占的字节数
    size_t stepImage = image.step;
    //cout<<stepImage<<endl;//1638，3通道图像的列数，将3个通道的像素合并为一个通道
    size_t stepImageGray = imageGray.step;
    //cout<<stepImageGray<<endl;//546，单通道图像的列数
    /*
    32位架构中普遍定义为：typedef unsigned int size_t;
    64位架构中普遍定义为：typedef  unsigned long size_t;
    */
    for (int i = 0; i < imageGray.rows; i++)
    {
        for (int j = 0; j < imageGray.cols; j++)
        {
            pointImageGray[i*stepImageGray + j] = (uchar)(0.114*pointImage[i*stepImage + 3 * j] + 0.587*pointImage[i*stepImage + 3 * j + 1] + 0.299*pointImage[i*stepImage + 3 * j + 2]);
        }
    }
}
 
 
//计算灰度图像的梯度图像模长
void SobelGradDirction(Mat &imageSource, Mat &imageSobelX, Mat &imageSobelY)
{
    imageSobelX = Mat::zeros(imageSource.size(), CV_32SC1);//32位signed int的单通道图
    imageSobelY = Mat::zeros(imageSource.size(), CV_32SC1);
    //cout<<imageSource.size()<<endl;//(546,300)
    /*
    CV_<bit_depth>(S|U|F)C<number_of_channels>, s for singed int; u for unsigned int; f for float;
    */
    //取出原图和X和Y梯度图的数组的首地址
    uchar *P = imageSource.data;
    uchar *PX = imageSobelX.data;
    uchar *PY = imageSobelY.data;
 
    //取imageSource和imageSobelX的列数
    size_t step = imageSource.step;
    //cout<<step<<endl;//546
    size_t stepXY = imageSobelX.step;
    //cout<<stepXY<<endl;//2184，是step的4倍，因为32位图是8位图深度的四倍，所以在列数上是4倍关系
    //int index = 0;//梯度方向角的索引
    for (int i = 1; i < imageSource.rows - 1; ++i)
    {
        for (int j = 1; j < imageSource.cols - 1; ++j)
        {
            //通过指针遍历图像上每一个像素
            double gradY = P[(i + 1)*step + j - 1] + P[(i + 1)*step + j] * 2 + P[(i + 1)*step + j + 1] - P[(i - 1)*step + j - 1] - P[(i - 1)*step + j] * 2 - P[(i - 1)*step + j + 1];
            PY[i*stepXY + j*(stepXY / step)] = abs(gradY);//梯度图像模长是正数，要取梯度的绝对值，并且每个值占32位，是8位图像的4倍，所以j*(stepXY / step)
 
            double gradX = P[(i - 1)*step + j + 1] + P[i*step + j + 1] * 2 + P[(i + 1)*step + j + 1] - P[(i - 1)*step + j - 1] - P[i*step + j - 1] * 2 - P[(i + 1)*step + j - 1];
            PX[i*stepXY + j*(stepXY / step)] = abs(gradX);
        }
    }
    //将32位的梯度数组转换成8位无符号整型
    convertScaleAbs(imageSobelX, imageSobelX);
    convertScaleAbs(imageSobelY, imageSobelY);
}
 
//对X方向上的梯度图像求平方
void SobelXX(const Mat imageGradX, Mat_<float> &SobelAmpXX)
{
    SobelAmpXX = Mat_<float>(imageGradX.size(), CV_32FC1);
    for (int i = 0; i < SobelAmpXX.rows; i++)
    {
        for (int j = 0; j < SobelAmpXX.cols; j++)
        {
            SobelAmpXX.at<float>(i, j) = imageGradX.at<uchar>(i, j)*imageGradX.at<uchar>(i, j);
        }
    }
    //convertScaleAbs(SobelAmpXX, SobelAmpXX);
}

//对Y方向上的梯度图像求平方
void SobelYY(const Mat imageGradY, Mat_<float> &SobelAmpYY)
{
    SobelAmpYY = Mat_<float>(imageGradY.size(), CV_32FC1);
    for (int i = 0; i < SobelAmpYY.rows; i++)
    {
        for (int j = 0; j < SobelAmpYY.cols; j++)
        {
            SobelAmpYY.at<float>(i, j) = imageGradY.at<uchar>(i, j)*imageGradY.at<uchar>(i, j);
        }
    }
    //convertScaleAbs(SobelAmpYY, SobelAmpYY);
}

//求X,Y方向上的梯度图像的乘积
void SobelXY(const Mat imageGradX, const Mat imageGradY, Mat_<float> &SobelAmpXY)
{
    SobelAmpXY = Mat_<float>(imageGradX.size(), CV_32FC1);
    for (int i = 0; i < SobelAmpXY.rows; i++)
    {
        for (int j = 0; j < SobelAmpXY.cols; j++)
        {
            SobelAmpXY.at<float>(i, j) = imageGradX.at<uchar>(i, j)*imageGradY.at<uchar>(i, j);
        }
    }
    //convertScaleAbs(SobelAmpXY, SobelAmpXY);
}

//计算二维高斯卷积核
double **getTwoGaussionArray(int size, double sigma)
{
    double sum = 0;
    //定义高斯核半径
    int kerR = size / 2;
    double **arr = new double*[size];
    for (int i=0; i<size; i++)
    {
        arr[i] = new double[size];
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0;j < size; j++)
        {
            //高斯函数前的常数可以不用计算，会在归一化的过程中给消去
            arr[i][j] = exp(-(pow(i - kerR,2)+pow(j - kerR,2))/2);
            sum += arr[i][j];//将所有的值进行相加
        }
    }
    //对高斯核的数值进行归一化
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j<size; j++)
        {
            arr[i][j] /= sum;
        }
    }
    return arr;
}

//计算响应函数矩阵
void harrisResponse(Mat_<float> &imageSobelXX, Mat_<float> &imageSobelYY, Mat_<float> &imageSobelXY, Mat_<float> &resultData, double **gaussian, float k, int rsize)
{
    //创建一张响应函数输出的矩阵,由于涉及高斯核卷积计算，这里不考虑边缘像素点是否为角点
    resultData = Mat_<float>(imageSobelXX.size(), CV_32FC1);
    for (int i = rsize; i < resultData.rows - rsize; i++)
    {
        for (int j = rsize; j < resultData.cols - rsize; j++)
        {
            float a = 0;//元素(i,j)对应的矩阵M的元素分别为a,c,c,b
            float b = 0;
            float c = 0;
            for (int m = i - rsize; m <= i + rsize; m++)
                for (int n = j - rsize; n <= j+rsize; n++)
                {
                    a += imageSobelXX.at<float>(m, n)*gaussian[m+rsize-i][n+rsize-j];
                    c += imageSobelYY.at<float>(m, n)*gaussian[m+rsize-i][n+rsize-j];
                    b += imageSobelXY.at<float>(m, n)*gaussian[m+rsize-i][n+rsize-j];
                }
            resultData.at<float>(i, j) = a*b - c*c - k*(a + b)*(a + b);
        }
    }
}
 
 
//非极大值抑制
void LocalMaxValue(Mat_<float> &resultData, Mat &srcGray, Mat &ResultImage, int rsize, int threshold)
{
    int r = rsize;//只考虑计算过卷积的像素
    bool flag = 1;//是否为极大值,默认为极大值
    ResultImage = srcGray.clone();
    for (int i = r; i < ResultImage.rows - r; i++)
    {
        for (int j = r; j < ResultImage.cols - r; j++)
        {
            //在3*3的领域内进行非极大值抑制
            flag = resultData.at<float>(i, j) > resultData.at<float>(i - 1, j - 1) &&
                   resultData.at<float>(i, j) > resultData.at<float>(i - 1, j) &&
                   resultData.at<float>(i, j) > resultData.at<float>(i - 1, j - 1) &&
                   resultData.at<float>(i, j) > resultData.at<float>(i - 1, j + 1) &&
                   resultData.at<float>(i, j) > resultData.at<float>(i, j - 1) &&
                   resultData.at<float>(i, j) > resultData.at<float>(i, j + 1) &&
                   resultData.at<float>(i, j) > resultData.at<float>(i + 1, j - 1) &&
                   resultData.at<float>(i, j) > resultData.at<float>(i + 1, j) &&
                   resultData.at<float>(i, j) > resultData.at<float>(i + 1, j + 1);
            if (flag)
            {
                if ((int)resultData.at<float>(i, j) > threshold)
                {
                    //此处必须要Point(j,i)，默认列索引在前面，4为线条粗细，Scalar为颜色
                    circle(ResultImage, Point(j, i), 4, Scalar(0, 0, 255), 2, 8, 0);
                }
            }
 
        }
    }
}

