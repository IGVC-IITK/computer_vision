// C++ Standard Library
#include <cassert>
#include <fstream>

// ROS
#include <ros/ros.h>
#include <ros/package.h>

// ROS Packages
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32MultiArray.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

// CUDA
#include <cuda_runtime_api.h>

// TensorRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "tensorrt/argsParser.h"
#include "tensorrt/logger.h"
#include "tensorrt/common.h"

using namespace nvinfer1;

samplesCommon::Args gArgs;

void printHelpInfo()
{
	std::cout << "Usage: ./fast_scnn_node [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
	std::cout << "--help          Display help information\n";
	std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (preprocessing/models)" << std::endl;
	std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
	std::cout << "--int8          Run in Int8 mode.\n";
	std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

// Callback Functions
void imageCallback(const sensor_msgs::ImageConstPtr& msg, IExecutionContext* context, 
					int in_channels, cv::Size in_spatial_dim, int num_classes, cv::Size out_spatial_dim, 
					const image_transport::Publisher& pub_seg, const ros::Publisher& pub_cf);

// TensorRT functions
bool onnxToTRTModel(const std::string& modelFile, unsigned int maxBatchSize, IHostMemory*& trtModelStream);
void doInference(IExecutionContext& context, float* input, int input_size, 
					float* output, int output_size, int batchSize);

int main(int argc, char **argv)
{
	// Processing input arguments
	bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
	if (!argsOK)
	{
		gLogError << "Invalid arguments" << std::endl;
		printHelpInfo();
		return EXIT_FAILURE;
	}
	if (gArgs.help)
	{
		printHelpInfo();
		return EXIT_SUCCESS;
	}
	if (gArgs.dataDirs.empty())
	{
		gArgs.dataDirs = std::vector<std::string>{ros::package::getPath("fast_scnn")+"/preprocessing/models/"};
	}

	// Initializing node
	ros::init(argc, argv, "fast_scnn_node");
	ros::NodeHandle nh;
	ros::Time t;

	// Getting parameters
	int in_channels, in_h, in_w;
	int num_classes, out_h, out_w; 
	bool overwrite_engine;
	nh.param("in_channels", in_channels, 3);
	nh.param("in_h", in_h, 720);
	nh.param("in_w", in_w, 1280);
	nh.param("num_classes", num_classes, 3);
	nh.param("out_h", out_h, 90);
	nh.param("out_w", out_w, 160);
	nh.param("overwrite_engine", overwrite_engine, false);
	cv::Size in_spatial_dim = cv::Size(in_w, in_h);
	cv::Size out_spatial_dim = cv::Size(out_w, out_h);

	// Reading TRT engine from disk
	std::ifstream file_in;
	int trtModelStreamSize;
	void* trtModelStreamData;
	bool performance_test = false;
	file_in.open(ros::package::getPath("fast_scnn")+"/preprocessing/models/fast_scnn.trt_engine", 
					std::ios::binary);
	if (file_in.good() && !overwrite_engine)
	{
		ROS_INFO_STREAM("Loading serialized TRT engine from disk.");
		file_in.seekg(0, file_in.end);
		trtModelStreamSize = file_in.tellg();
		trtModelStreamData = malloc(trtModelStreamSize);
		file_in.seekg (0, file_in.beg);
		file_in.read((char*)trtModelStreamData, trtModelStreamSize);
		file_in.close();
	}
	else
	{
		ROS_INFO_STREAM("Generating new TRT engine from ONNX model...");
		IHostMemory* trtModelStream{nullptr};
		onnxToTRTModel("fast_scnn.onnx", 1, trtModelStream);
		assert(trtModelStream != nullptr);

		trtModelStreamSize = trtModelStream->size();
		trtModelStreamData = malloc(trtModelStreamSize);
		std::memcpy(trtModelStreamData, trtModelStream->data(), trtModelStreamSize);
		trtModelStream->destroy();

		ROS_INFO_STREAM("Saving serialized TRT engine to disk.");
		std::ofstream file_out;
		file_out.open(ros::package::getPath("fast_scnn")+"/preprocessing/models/fast_scnn.trt_engine", 
						std::ios::binary);
		file_out.write((char*)trtModelStreamData, trtModelStreamSize);
		file_out.close();
		performance_test = true;
	}
	
	// Deserializing the engine
	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	if (gArgs.useDLACore >= 0)
	{
		runtime->setDLACore(gArgs.useDLACore);
	}

	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStreamData, trtModelStreamSize, nullptr);
	assert(engine != nullptr);
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);

	// Performing a performance test if a new engine has been generated
	if (performance_test)
	{
		ROS_INFO_STREAM("Performance Test... (15 seconds)");
		int num_frames = 0;
		t = ros::Time::now();
		while ((ros::Time::now()-t).toSec() < 15.0)
		{
			// Generating dummy input
			float* network_input = new float[in_channels*in_spatial_dim.area()];
				for (int i = 0; i < in_channels*in_spatial_dim.area(); i++)
					network_input[i] = 1.0;
			// Performing inference
			float* network_output = new float[num_classes*out_spatial_dim.area()];
			doInference(*context, network_input, in_channels*in_spatial_dim.area(), 
						network_output, num_classes*out_spatial_dim.area(), 1);
			delete [] network_input;
			delete [] network_output;
			num_frames++;
		}
		ROS_INFO_STREAM("Average FPS: "<<((float)num_frames)/(ros::Time::now() - t).toSec());
	}

	// Creating subscriber and publisher objects
	image_transport::ImageTransport it(nh);
	image_transport::Publisher pub_seg = it.advertise("fast_scnn/segmented_image", 1);
	ros::Publisher pub_cf = nh.advertise<std_msgs::Float32MultiArray>("/fast_scnn/class_fractions", 1);
	image_transport::Subscriber sub_img = it.subscribe("/image", 1, 
		boost::bind(imageCallback, _1, context, 
					in_channels, in_spatial_dim, num_classes, out_spatial_dim,
					boost::ref(pub_seg), boost::ref(pub_cf)));

	// Performing inference on incoming image messages
	ros::spin();

	// Destroying the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	return 0;
}

////////////////////////
// Callback functions //
////////////////////////

void imageCallback(const sensor_msgs::ImageConstPtr& msg, IExecutionContext* context, 
					int in_channels, cv::Size in_spatial_dim, int num_classes, cv::Size out_spatial_dim, 
					const image_transport::Publisher& pub_seg, const ros::Publisher& pub_cf)
{
	// Processing the image message
	std::string image_format;
	cv::Mat image_in;
	if (msg->encoding == sensor_msgs::image_encodings::BGR8 || 
		msg->encoding == sensor_msgs::image_encodings::RGB8)
	{
		if (in_channels == 3)
			image_format = sensor_msgs::image_encodings::RGB8; // ASSUMING THAT THE NETWORK EXPECTS RGB IMAGES
		else if (in_channels == 1)
		{
			image_format = sensor_msgs::image_encodings::MONO8;
			ROS_WARN_STREAM_ONCE("Converting color image to grayscale.");
		}
		else
		{
			ROS_ERROR_STREAM("fast_scnn_node expects in_channels = 1 or 3.");
			return;
		}
		
	}
	else if (msg->encoding == sensor_msgs::image_encodings::MONO8)
	{
		if (in_channels == 1)
			image_format = sensor_msgs::image_encodings::MONO8;
		else if (in_channels == 3)
		{
			image_format = sensor_msgs::image_encodings::RGB8;
			ROS_ERROR("Should not convert from '%s' to '%s'.", msg->encoding.c_str(), image_format.c_str());
			return;
		}
		else
		{
			ROS_ERROR_STREAM("fast_scnn_node expects in_channels = 1 or 3.");
			return;
		}
	}
	else
	{
		ROS_ERROR_STREAM("Unexpected input image format.");
		return;
	}

	// Preparing flat input for network
	try
	{
		image_in = cv_bridge::toCvShare(msg, image_format)->image.clone();
		if(!image_in.empty())
		{
			if (!(image_in.size().height == in_spatial_dim.height && 
					image_in.size().width == in_spatial_dim.width))
				cv::resize(image_in, image_in, in_spatial_dim);
		}
		else
		{
			ROS_ERROR_STREAM("Empty image received.");
			return;
		}
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to '%s'.", msg->encoding.c_str(), image_format.c_str());
		return;
	}
	float* network_input = new float[in_channels*in_spatial_dim.area()];
	for (int y = 0; y < in_spatial_dim.height; y++)
		for (int x = 0; x < in_spatial_dim.width; x++)
			for (int current_channel = 0; current_channel < in_channels; current_channel++)
			{
				// Re-ordering, conversion to float and normalization
				network_input[current_channel*in_spatial_dim.area() + y*in_spatial_dim.width + x] = 
				(((float)(image_in.at<cv::Vec3b>(y, x)[current_channel]))/255.0 - 0.5)*2.0;
			}

	// Performing inference
	float* network_output = new float[num_classes*out_spatial_dim.area()];
	doInference(*context, network_input, in_channels*in_spatial_dim.area(), 
				network_output, num_classes*out_spatial_dim.area(), 1);
	delete [] network_input;

	// Converting flat output from network to grayscale image
	cv::Mat image_out(out_spatial_dim, CV_8UC1);
	std::vector<int> class_counts(num_classes, 0);
	for (int y = 0; y < out_spatial_dim.height; y++)
		for (int x = 0; x < out_spatial_dim.width; x++)
		{
			float max_prob = network_output[0*out_spatial_dim.area() + y*out_spatial_dim.width + x];
			int max_prob_class = 0;
			for (int current_class = 1; current_class < num_classes; current_class++)
			{
				float prob = network_output[current_class*out_spatial_dim.area() + y*out_spatial_dim.width + x];
				if (max_prob < prob)
				{
					max_prob = prob;
					max_prob_class = current_class;
				}
			}
			image_out.at<uchar>(y, x) = (255/(num_classes-1))*max_prob_class;
			class_counts[max_prob_class]++;
		}
	delete [] network_output;

	// Publishing segmented image
	std_msgs::Header image_header = msg->header;
	image_format = sensor_msgs::image_encodings::MONO8;
	sensor_msgs::ImagePtr segmented_msg = cv_bridge::CvImage(image_header, image_format.c_str(), image_out).toImageMsg();
	pub_seg.publish(segmented_msg);

	// Publishing class fractions
	std_msgs::Float32MultiArray class_fractions;
	class_fractions.layout.dim.resize(1);
	class_fractions.layout.dim[0].label = "semantic_class";
	class_fractions.layout.dim[0].size = 3;
	class_fractions.layout.dim[0].stride = 3;
	class_fractions.layout.data_offset = 0;
	class_fractions.data.resize(num_classes);
	for (int current_class = 0; current_class < num_classes; current_class++)
	{
		class_fractions.data[current_class] = 
			((float)class_counts[current_class])/((float)out_spatial_dim.area());
	}
	pub_cf.publish(class_fractions);
}

/////////////////////////////////////////////////////////////////////////
// TensorRT functions from NVIDIA TensorRT Samples (slightly modified) //
/////////////////////////////////////////////////////////////////////////

bool onnxToTRTModel(const std::string& modelFile,	// name of the onnx model
					unsigned int maxBatchSize,		// must be at least as large as the required batch size
					IHostMemory*& trtModelStream)	// output buffer for the TensorRT model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
	assert(builder != nullptr);
	INetworkDefinition* network = builder->createNetwork();

	auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

	if (!parser->parseFromFile(locateFile(modelFile, gArgs.dataDirs).c_str(),
								static_cast<int>(gLogger.getReportableSeverity())))
	{
		gLogError<<"Failure while parsing ONNX file"<<std::endl;
		return false;
	}

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	// Optional optimizations
	builder->setFp16Mode(gArgs.runInFp16);
	builder->setInt8Mode(gArgs.runInInt8);
	if (gArgs.runInInt8)
	{
		samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
	}
	samplesCommon::enableDLA(builder, gArgs.useDLACore);
	
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we can destroy the parser
	parser->destroy();

	// serialize the engine, then close everything down
	trtModelStream = engine->serialize();
	engine->destroy();
	network->destroy();
	builder->destroy();

	return true;
}

void doInference(IExecutionContext& context, float* input, int input_size, 
					float* output, int output_size, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly
	// IEngine::getNbBindings() of these, but in this case we know that there is exactly one input and
	// one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex{}, outputIndex{};
	for (int b = 0; b < engine.getNbBindings(); ++b)
	{
		if (engine.bindingIsInput(b))
			inputIndex = b;
		else
			outputIndex = b;
	}

	// The input and output sizes must stay constant.
	static const int input_size_ = input_size;
	static const int output_size_ = output_size;

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize*input_size_*sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize*output_size_*sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize*input_size_*sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize*output_size_*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}
