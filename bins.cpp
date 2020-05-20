/** @file bins.cpp
 *
 *  @brief Vision functions to detect the bins
 *
 *  @author David Zhang
 *  @author Craig Wang
 */
#include <ros/ros.h>
#include "vision/service.hpp"
#include "vision/filters.hpp"
#include "vision/log.hpp"


Observation VisionService::findBins(const cv::Mat &input)
{
	/*
	 * This bins code is meant to be run at Suhas' pool, with a black outline
	 * and white inside. Do not use for competition.
	 */
	// Illuminate image using filter.
	// cv::Mat illum = illumination(input);
	cv::Mat illum = input;

	// Strong blur to remove noise from image.
	cv::Mat blur;
	cv::blur(illum, blur, cv::Size(9, 9));
	log(illum, 'e');

	// Threshold for black.
	cv::Mat thresh;
	cv::Mat cdst;
	cv::inRange(blur, cv::Scalar(0, 0, 0), cv::Scalar(46, 36, 36), thresh);
	cv::cvtColor(thresh, cdst, cv::COLOR_GRAY2BGR);
	// cv::cvtColor(thresh, thresh, cv::COLOR_BGR2GRAY);

	// Contour detection.
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(thresh, contours, hierarchy, CV_RETR_TREE, 
			CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	for (int i = 0; i < contours.size(); i++)
	{
		cv::drawContours(cdst, contours, i, cv::Scalar(0, 255, 0), 2, 8, 
				hierarchy, 0, cv::Point());
	}

	// Approximate and convert to rectangles.
	std::vector<std::vector<cv::Point>> contour_polygons (contours.size());
	std::vector<cv::Rect> rectangles (contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), contour_polygons[i], 
				0.01*cv::arcLength(cv::Mat(contours[i]), true), true);
		rectangles[i] = cv::boundingRect(cv::Mat(contour_polygons[i]));
	}
	for (int i = 0; i < contours.size(); i++)
	{
		cv::drawContours(cdst, contour_polygons, i, cv::Scalar(255, 0, 0), 1, 8, 
				std::vector<cv::Vec4i>(), 0, cv::Point());
		cv::rectangle(cdst, rectangles[i].tl(), rectangles[i].br(), 
				cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	// Choose largest rectangle that is in an appropriate ratio.
	std::sort(rectangles.begin(), rectangles.end(), [](const cv::Rect &a, 
				const cv::Rect &b) -> bool { return a.area() > b.area(); });
	for (int i = 0; i < rectangles.size(); i++)
	{
		float height = rectangles[i].height;
		float width = rectangles[i].width;
		float rect_ratio = height/width;
		if (rect_ratio < 2. && rect_ratio > 0.5 && 
				rectangles[i].tl().x+width/2. > 150 && 
				rectangles[i].br().x+height/2. < 1138)
		{
			cv::rectangle(cdst, rectangles[i].tl(), rectangles[i].br(), 
					cv::Scalar(255, 0, 255), 3, 8, 0);		
			int x = (rectangles[i].tl().x+rectangles[i].br().x)/2;
			int y = (rectangles[i].tl().y+rectangles[i].br().y)/2;
			cv::circle(cdst, cv::Point(x, y), 4, cv::Scalar(255, 0, 255), 3);
			log(cdst, 'e');
			return Observation(0.8, y, x, 0);
		}
	}

	// No contours found. I doubt this will run often because the sub is more
	// prone to picking up noise instead of nothing at all.
	return Observation(0, 0, 0, 0);
}

Observation VisionService::findBinsML(cv::Mat img)
{
	ROS_INFO("Starting machine learning detection for BIN.");
	log(img, 'd');

	//IMPORTANT: Edit architecture of model (eg. efficientdet-d3 -> phi=3)
	int phi = 0;
	int image_sizes[] = {512, 640, 768, 896, 1024, 1152, 1280};      //image sizes that effdet uses
	int image_size = image_sizes[phi];                               //takes the image size that our model uses
	std::string classes[] = {"bin"};                                 //list of classes
	std::vector<cv::Scalar> colors = {{0, 255, 255}};                //setup our box color (blue,green,red) (this is yellow)

	// Inititialize the model's input and output tensors
	auto inpName = new Tensor(model, "input_1");
	auto out_boxes = new Tensor(model, "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3");
	auto out_scores = new Tensor(model, "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3");
	auto out_labels = new Tensor(model, "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3");

	// Process image
	cv::Mat inp;                                                //model input image
	float scale = resize(img, image_size);                      //downsize for model compatibility, scale factor is for resizing boxes back to og image later
	underwaterEnhance(img);                                     //phoebe enhance on small img to avoid time complexity
	cv::cvtColor(img, inp, CV_BGR2RGB);                         //copy and convert from bgr to rgb
	std::vector<float> img_data = preprocess(inp, image_size);  //normalize image for model input

	// Put data in tensor.
	inpName->set_data(img_data, {1, image_size, image_size, 3});

	// Run model
	model.run(inpName, { out_boxes, out_scores, out_labels });

	// Convert tensors to std vectors
	auto boxes = out_boxes->get_data<float>();
	auto scores = out_scores->get_data<float>();
	auto labels = out_labels->get_data<int>();

	// Iterate over results and draw the boxes for visualization!
	for (int i=0; i<scores.size(); i++) {
		if (scores[i] > 0.3) {
			// Extract output values
			float score = scores[i];
			int label = labels[i];
			float xmin = boxes[i*4];    //the boxes come in 4 values: [xmin (left), ymin (top), xmax (right), ymax (bottom)]
			float ymin = boxes[i*4+1];
			float xmax = boxes[i*4+2];
			float ymax = boxes[i*4+3];
			if (label == 0)
				ROS_INFO("Bin Found");

			// Aesthetically visualize output box, label, and score
			drawBox(img, xmin, ymin, xmax, ymax, classes, label, score, colors);
			//logs image
			log(img, 'e');                                                                       

			// Change bbox values back to original image to calculate accurate angles
			xmin /= scale; 
			ymin /= scale;
			xmax /= scale;
			ymax /= scale;
			// Returns observation(score, ycenter, xcenter, distance)
			return Observation(score, (ymin+ymax)/2, (xmin+xmax)/2, 0);                              
		}
		else {            //since the outputs are already sorted by score
			break;        //if it's lower than the thres then we know it's the last highest one,
		}                 //so we can afford to break out because all the other outputs will be below the threshold
	}

	return Observation(0, 0, 0, 0);
}
