// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

//This file can be used to apply hysteresis thresholding on the image obtained after segmentation and averaging!

#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"
#include <queue>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "./gSLICr_Lib/objects/gSLICr_spixel_info.h"
#include "./gSLICr_Lib/engines/gSLICr_seg_engine_shared.h"

using namespace std;
using namespace cv;

typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;

typedef struct {
    double h;       // angle in degrees
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1
} hsv;

static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0              
        // s = 0, h is undefined
        out.s = 0.0;
        out.h = NAN;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}
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



const int Slider_max = 100;
int Lvalue_slider, Hvalue_slider,Value_slider;
double Lvalue,Hvalue,Value;
void on_trackbar( int, void* )
{
    Lvalue = (double) Lvalue_slider/Slider_max ;
}
void on_trackbar2( int, void* )
{
    Hvalue = (double) Hvalue_slider/Slider_max ;
}
void on_trackbar3( int, void* )
{
    Value = (double) Value_slider/Slider_max ;
}
std::string ToString(int val)
{
    stringstream ss;
    ss<<val;
    return ss.str();
}



int main()
{
    /*VideoCapture cap("../sam.webm");


    if (!cap.isOpened()) 
    {
        cerr << "unable to open camera!\n";
        return -1;
    }*/
    
    // gSLICr settings
    gSLICr::objects::settings my_settings;

    my_settings.img_size.x = 500;
    my_settings.img_size.y = 500;
    my_settings.no_segs = 250;
    my_settings.spixel_size = 100;
    my_settings.coh_weight = 1.0f;
    my_settings.no_iters = 50;
    my_settings.color_space = gSLICr::CIELAB; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
    my_settings.seg_method = gSLICr::GIVEN_NUM; // or gSLICr::GIVEN_NUM for given number
    my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step
    int n = int(sqrt(my_settings.no_segs));

    // instantiate a core_engine
    gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);
    gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
    gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);


    Size s(my_settings.img_size.x, my_settings.img_size.y);
    Size s1(640, 480);

    Mat oldFrame, frame;
    Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);
    
    //Initiating Slider Launch
    Lvalue_slider = 0;
    Hvalue_slider = 0;
    Value_slider = 0;
    namedWindow("Adjuster", 1);
    char TrackbarName[50], TrackbarName2[50], TrackbarName3[50];
    sprintf( TrackbarName, "Lvalue x %d", Slider_max );
    sprintf( TrackbarName2, "Hvalue x %d", Slider_max );
    sprintf( TrackbarName3, "Value x %d", Slider_max );
    createTrackbar( TrackbarName, "Adjuster", &Lvalue_slider, Slider_max, on_trackbar );
    createTrackbar( TrackbarName2, "Adjuster", &Hvalue_slider, Slider_max, on_trackbar2 );
    createTrackbar( TrackbarName3, "Adjuster", &Value_slider, Slider_max, on_trackbar3 );
    on_trackbar( Lvalue_slider, 0 );
    on_trackbar2(Hvalue_slider,0);
    on_trackbar3(Value_slider,0);
    double saturation=0.50;
    int key, h=0;
    cin>>h;
                while(1){
   
    //for(int i=0;i<=h;i++)
    //{
        //h++;
        std::string first ("../dr/");
        std::string sec (".png");
        std::string mid = ToString(h);
        std::string name;
        name=first+mid+sec;
        oldFrame = cv::imread(name);
        //cout<<Lvalue<<"+"<<Hvalue<<"-\n";
        
        float blue_sum[my_settings.no_segs*(2)] = {0};
        float green_sum[my_settings.no_segs*(2)] = {0};
        float red_sum[my_settings.no_segs*(2)] = {0};
        int blue[my_settings.img_size.x][my_settings.img_size.y] = {0};
        int green[my_settings.img_size.x][my_settings.img_size.y] = {0};
        int red[my_settings.img_size.x][my_settings.img_size.y] = {0};
        int sum_x[my_settings.no_segs] = {0};
        int sum_y[my_settings.no_segs] = {0};
        int matrix[250000] = {0};
        int count[250]={0};
        
        pub_info *obj=(pub_info*)malloc(2*my_settings.no_segs*sizeof(pub_info));
        resize(oldFrame, frame, s);
        
        load_image(frame, in_img);
        gSLICr_engine->Process_Frame(in_img);
        gSLICr_engine->Draw_Segmentation_Result(out_img);
        load_image(out_img, boundry_draw_frame);
        //cv::namedWindow("InitialImg",0);
        //cv::imshow("InitialImg", frame);
        //cv::namedWindow("FinalImg",0);
        
        gSLICr_engine->Write_Seg_Res_To_PGM("abc",matrix);
        Mat M = frame;
        Mat M2,Mimg;
        int lable;
        int prev_lable[my_settings.no_segs]={0};

        ///Retrieving colour channels from the jimage
        for(int i=0;i<my_settings.img_size.x;i++)
        {
            for(int j=0;j<my_settings.img_size.y;j++)
            {
                blue[i][j] = M.at<cv::Vec3b>(i,j)[0]; // b
                green[i][j] = M.at<cv::Vec3b>(i,j)[1]; // g
                red[i][j] = M.at<cv::Vec3b>(i,j)[2]; // r
            }
        }

        ///Summing over the pixel Lvalues of all segments
        for(int i=0;i<my_settings.img_size.x*my_settings.img_size.y;i++)
        {
            blue_sum[matrix[i]]+= blue[i/my_settings.img_size.x][i%my_settings.img_size.x];
            red_sum[matrix[i]]+= red[i/my_settings.img_size.x][i%my_settings.img_size.x];
            green_sum[matrix[i]]+= green[i/my_settings.img_size.x][i%my_settings.img_size.x];
            sum_x[matrix[i]]+=  ( i/my_settings.img_size.x);
            sum_y[matrix[i]]+= (i%my_settings.img_size.x);
            count[matrix[i]]++;
        }

        queue <int> clrbox;
        for(int i=0;i<my_settings.no_segs;i++)
        {

            rgb rgb_obj;
            rgb_obj.r = (red_sum[i]/count[i]) ;// r
            rgb_obj.g = (green_sum[i]/count[i]) ;// r
            rgb_obj.b = (blue_sum[i]/count[i]) ;// r
            hsv hsv_obj= rgb2hsv(rgb_obj);
            if(hsv_obj.h>Value*360){
                if((hsv_obj.v>Lvalue*255 && hsv_obj.v<Hvalue*255)){
                    prev_lable[i]=1;

            }
            else if( hsv_obj.v>Hvalue*255)
                {
                    clrbox.push(i);
                    prev_lable[i]=2;
                }
            else
                prev_lable[i]=0;
            }
            else prev_lable[i]=0;

            //cout<<"("<<i<<") "<<int(hsv_obj.h)<<" "<<int(hsv_obj.s)<<" "<<int(hsv_obj.v)<<" "<<lable<<"\n";

        }
        while(!clrbox.empty())
        {
            int var = clrbox.front();
            clrbox.pop();
            prev_lable[var]=3;
            int x = (int)(var/n);
            int y = (int)(var%n);
            if(n*(x-1) + y-1>=0|| y!=0 ){
                if(prev_lable[n*(x-1) + y-1]==1){
                    prev_lable[n*(x-1) + y-1]=2;
                    clrbox.push(n*(x-1) + y-1);
                }
            }
            if(n*(x-1) + y>=0)
            {
                if(prev_lable[n*(x-1) + y]==1){
                    prev_lable[n*(x-1) + y]=2;
                    clrbox.push(n*(x-1) + y);
                }
            }
            if(n*(x-1) + y + 1>=0 && y!=n-1){
                if(prev_lable[n*(x-1) + y + 1]==1){
                    prev_lable[n*(x-1) + y + 1]=2;
                    clrbox.push(n*(x-1) + y + 1);
                }}
            if(var-1>=0 && y!=0){
                if(prev_lable[var-1]==1){
                    prev_lable[var-1]=2;
                    clrbox.push(var-1);
                }}
            if(y!=n-1){
                if(prev_lable[var+1]==1){
                    prev_lable[var+1]=2;
                    clrbox.push(var+1);
                }}
            if(n*(x+1) + y -1<n && y!=0){
                if(prev_lable[n*(x+1) + y -1]==1){
                    prev_lable[n*(x+1) + y -1]=2;
                    clrbox.push(n*(x+1) + y -1);
                } }
            if(n*(x+1) + y<n){
                if(prev_lable[n*(x+1) + y]==1){
                    prev_lable[n*(x+1) + y]=2;
                    clrbox.push(n*(x+1) + y);
                }}
            if(n*(x+1) + y+1<n && y!=n-1){
                if(prev_lable[n*(x+1) + y + 1]==1){
                    prev_lable[n*(x+1) + y + 1]=2;
                    clrbox.push(n*(x+1) + y + 1);
                    }
            }

        }
        for(int i=0;i<my_settings.no_segs;i++)
        {
            rgb rgb_obj;
            rgb_obj.r = (red_sum[i]/count[i]) ;// r
            rgb_obj.g = (green_sum[i]/count[i]) ;// g
            rgb_obj.b = (blue_sum[i]/count[i]) ;// b
            if(prev_lable[i]==3)
                lable=1;
            else
                lable=0;

            red_sum[i] = lable*255*count[i];
            green_sum[i] = lable*255*count[i];
            blue_sum[i] = lable*255*count[i];
            
            /*----------- Uncomment to print rgb values of each super-pixel along with black(0)/white(1) lable-----------*/           
            
            //if(int(rgb_obj.r)> 0)
            //    cout<<"("<<i<<") "<<int(rgb_obj.r)<<" "<<int(rgb_obj.g)<<" "<<int(rgb_obj.b)<<" "<<lable<<"\n";


        }


        ///Re-inserting the average Lvalues back into the image
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
        
        Canny( M, M2, 100, 180, 3);
        Mat M3;
        resize(M, M3, s1);
        //cv::imshow("FinalImg",M2);
        cv::namedWindow("Binary",5);
        cv::imshow("Binary",M3);

        free(obj);

        key = (char)waitKey(1);
        if (key == 27) break;
        
    }
    destroyAllWindows();
    return 0;
}