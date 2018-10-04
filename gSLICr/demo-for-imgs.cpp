// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr
/*
    This file produces the segmented image after averaging.
*/
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "./gSLICr_Lib/objects/gSLICr_spixel_info.h"
#include "./gSLICr_Lib/engines/gSLICr_seg_engine_shared.h"

using namespace std;
using namespace cv;

typedef struct RgbColor
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} RgbColor;

typedef struct HsvColor
{
    unsigned char h;
    unsigned char s;
    unsigned char v;
} HsvColor;

struct pub_info
{
    float centre_x;
    float centre_y;
    float avg_colors[3];
    float size;
};

void load_image(const Mat& inimg, gSLICr::UChar4Image* outimg)
{
    gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

    for (int y = 0; y < outimg->noDims.y;y++)
        for (int x = 0; x < outimg->noDims.x; x++)
        {
            int idx = x + y * outimg->noDims.x;
            outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
            outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
            outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
        }
}

void load_image(const gSLICr::UChar4Image* inimg, Mat& outimg)
{
    const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

    for (int y = 0; y < inimg->noDims.y; y++)
        for (int x = 0; x < inimg->noDims.x; x++)
        {
            int idx = x + y * inimg->noDims.x;
            outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
            outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
            outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
        }
}

HsvColor RgbToHsv(RgbColor rgb)
{
    HsvColor hsv;
    unsigned char rgbMin, rgbMax;

    rgbMin = rgb.r < rgb.g ? (rgb.r < rgb.b ? rgb.r : rgb.b) : (rgb.g < rgb.b ? rgb.g : rgb.b);
    rgbMax = rgb.r > rgb.g ? (rgb.r > rgb.b ? rgb.r : rgb.b) : (rgb.g > rgb.b ? rgb.g : rgb.b);

    hsv.v = rgbMax;
    if (hsv.v == 0)
    {
        hsv.h = 0;
        hsv.s = 0;
        return hsv;
    }

    hsv.s = 255 * long(rgbMax - rgbMin) / hsv.v;
    if (hsv.s == 0)
    {
        hsv.h = 0;
        return hsv;
    }

    if (rgbMax == rgb.r)
        hsv.h = 0 + 43 * (rgb.g - rgb.b) / (rgbMax - rgbMin);
    else if (rgbMax == rgb.g)
        hsv.h = 85 + 43 * (rgb.b - rgb.r) / (rgbMax - rgbMin);
    else
        hsv.h = 171 + 43 * (rgb.r - rgb.g) / (rgbMax - rgbMin);

    return hsv;
}

std::string ToString(int val)
{
    stringstream ss;
    ss<<val;
    return ss.str();
}

int main()
{
    gSLICr::objects::settings my_settings;

    my_settings.img_size.x = 500;
    my_settings.img_size.y = 500;
    my_settings.no_segs = 250;
    my_settings.spixel_size = 200;
    my_settings.coh_weight = 1.0f;
    my_settings.no_iters = 5;
    my_settings.color_space = gSLICr::CIELAB; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
    my_settings.seg_method = gSLICr::GIVEN_NUM; // or gSLICr::GIVEN_NUM for given number
    my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step


    // instantiate a core_engine
    gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);
    gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
    gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);


    Size s(my_settings.img_size.x, my_settings.img_size.y);
    Size s1(640, 480);

    Mat oldFrame, frame;
    Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);
    
    
    int key, h=0;
    cin>>h;
        
    while(1)
    //for(int i=0;i<=h;i++)
    {
        //h++;
        std::string first ("../dr/");
        std::string sec (".png");
        std::string mid = ToString(h);
        std::string name,newname;
        name=first+mid+sec;
        newname=mid+sec;
        oldFrame = cv::imread(name);
        
        float blue_sum[my_settings.no_segs*(2)] = {0};
        float green_sum[my_settings.no_segs*(2)] = {0};
        float red_sum[my_settings.no_segs*(2)] = {0};
        int blue[my_settings.img_size.x][my_settings.img_size.y] = {0};
        int green[my_settings.img_size.x][my_settings.img_size.y] = {0};
        int red[my_settings.img_size.x][my_settings.img_size.y] = {0};
        int sum_x[my_settings.no_segs] = {0};
        int sum_y[my_settings.no_segs] = {0};
        int matrix[250000] = {0};
        int count[200]={0};
        
        pub_info *obj=(pub_info*)malloc(2*my_settings.no_segs*sizeof(pub_info));
        resize(oldFrame, frame, s);
        
        load_image(frame, in_img);
        gSLICr_engine->Process_Frame(in_img);
        gSLICr_engine->Draw_Segmentation_Result(out_img);
        load_image(out_img, boundry_draw_frame);
        //cv::namedWindow("tt",0);
        //cv::imshow("tt", boundry_draw_frame);
        cv::namedWindow("tt2",0);
        
        gSLICr_engine->Write_Seg_Res_To_PGM("abc",matrix);
        Mat M = frame;
        Mat M2;
        int lable;

        ///Retrieving colour channels from the image
        for(int i=0;i<my_settings.img_size.x;i++)
        {
            for(int j=0;j<my_settings.img_size.y;j++)
            {
                blue[i][j] = M.at<cv::Vec3b>(i,j)[0]; // b
                green[i][j] = M.at<cv::Vec3b>(i,j)[1]; // g
                red[i][j] = M.at<cv::Vec3b>(i,j)[2]; // r
            }
        }

        ///Summing over the pixel values of all segments
        for(int i=0;i<my_settings.img_size.x*my_settings.img_size.y;i++)
        {
            blue_sum[matrix[i]]+= blue[i/my_settings.img_size.x][i%my_settings.img_size.x];
            red_sum[matrix[i]]+= red[i/my_settings.img_size.x][i%my_settings.img_size.x];
            green_sum[matrix[i]]+= green[i/my_settings.img_size.x][i%my_settings.img_size.x];
            sum_x[matrix[i]]+=  ( i/my_settings.img_size.x);
            sum_y[matrix[i]]+= (i%my_settings.img_size.x);
            count[matrix[i]]++;
        }

        ///Re-inserting the average values back into the image
        for(int i=0;i<my_settings.img_size.y;i++)
        {
            for(int j=0;j<my_settings.img_size.x;j++)
            {
                M.at<cv::Vec3b>(i,j)[0] = blue_sum[matrix[i*my_settings.img_size.x + j ]]/count[matrix[i*my_settings.img_size.x + j]] ;// b
                M.at<cv::Vec3b>(i,j)[1] = green_sum[matrix[i*my_settings.img_size.x + j ]]/count[matrix[i*my_settings.img_size.x + j]] ;// g
                M.at<cv::Vec3b>(i,j)[2] = red_sum[matrix[i*my_settings.img_size.x + j ]]/count[matrix[i*my_settings.img_size.x + j]] ;// r
            }
        }

        /// now compute all properties of pub_info obj[no_segs]
        for(int i=0;i<=my_settings.no_segs;i++)
        {
            if(count[i]!=0)
            {
                obj[i].size=count[i];
                obj[i].avg_colors[0] = (int)(blue_sum[i]/count[i]);
                obj[i].avg_colors[1] = (int)(green_sum[i]/count[i]);
                obj[i].avg_colors[2] = (int)(red_sum[i]/count[i]);
                obj[i].centre_x = (int)(sum_x[i]/count[i]);
                obj[i].centre_y = (int)(sum_y[i]/count[i]);
            }
        }
        
        resize(M, M2, s1);
        cv::imshow("tt2",M2);

        cv::imwrite(newname,M2);



        key = (char)waitKey(1);
        if (key == 27) break;
        free(obj);
      
      
    }


    destroyAllWindows();
    return 0;
}