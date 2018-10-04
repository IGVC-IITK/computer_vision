#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp> 
using namespace :: std;
using namespace cv;
using namespace cv::ml;
int main ()
{
	int lines = 135094;
	int features = 75;
	Mat_<float> data(lines, features);
	Mat_<float> responses(data.rows, 1);
	int label;
	ifstream read("./train.txt");

	for(int i=0;i<lines;i++)
	{
		for (int j=0;j<features;j++)
		{
			read>>data(i,j);
		}

		read>>label;

		if(label==1)
			responses(i,0)=1;
		
		if(label==0)
			responses(i,0)=-1;
	}

    Mat_<int> layerSizes(1, 5);
    layerSizes(0, 0) = data.cols;
    layerSizes(0, 1) = 50;
    layerSizes(0, 2) = 30;
    layerSizes(0, 3) = 15;
    layerSizes(0, 4) = responses.cols;

    Ptr<ANN_MLP> mlp = ANN_MLP::create();
    mlp->setLayerSizes(layerSizes);
    mlp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER +TermCriteria::EPS,10000,0.0001));
    //mlp->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.5, 1);
    mlp->setActivationFunction(ANN_MLP::RELU, 0.5, 3);
    mlp->setTrainMethod(ANN_MLP::BACKPROP,0.1,0.1);
    Ptr<TrainData> trainData = TrainData::create(data, ROW_SAMPLE, responses);

    mlp->train(trainData);

    //cv::FileStorage fs("mlp.yml", cv::FileStorage::WRITE);
    //mlp->write(*fs,"mlp");
    mlp->save("home/ryuzakii/catkin_ws/src/classifier/src/mlp5layer-relu.yml");

    if (mlp->isTrained())
    {
        cout<<"Yo!\n";
        Mat result;
        for (int i=0; i<data.rows; ++i)
        {
            mlp->predict(data.row(i), result);
            cout << result << endl;
        }
    }
return 0;
}
