// Probably don't need to include these things because the sub's script already has it in its header
#include "/content/cppflow/include/Model.h"  //cppflow stuff
#include "/content/cppflow/include/Tensor.h"

#include "opencv2/opencv.hpp"

#include <iostream>

// helper function that resizes the image, processes the image, edits the scale as an output, and returns img_data std vector ready for model input
std::vector<float> preprocess(cv::Mat image, float &scale, int image_size=512)
{
    image.convertTo(image, CV_32FC3);            //converts to float matrix so we can multiply and divide

    int image_height = image.rows;     //initialize image_height variable for convenience
    int image_width = image.cols;      //initialize image_width variable for convenience
    int resized_height, resized_width;    //declare resized_height, resized_width variables to use in the if/else statements

    // calculates what height and width to resize to preserve height/width ratios
    if (image_height > image_width)
    {                 //if the image more tall than wide
        scale = (float)image_size / image_height;     //then the taller side will become the image_size (eg. 512), so the scale is 512/og_h
        resized_height = image_size;                  //downsize taller side to 512
        resized_width = (int)(image_width * scale);   //scale the width down by same scale as the height, preserving side ratios
    }
    else
    {                                                 //otherwise, this means the image is more wide than tall
        scale = (float)image_size / image_width;      //repeat same procedures as above except for width
        resized_height = (int)(image_height * scale);
        resized_width = image_size;
    }

    // cv matrix process stuff
    cv::resize(image, image, cv::Size(resized_width, resized_height));         //resize the image, keeping ratios
    cv::Mat temp(image_size, image_size, CV_32FC3, cv::Scalar(128,128,128));   // makes temporary mat with shape (512,512,3) filled with 128s
    image.copyTo(temp(cv::Rect(0, 0, image.cols, image.rows)));                // pastes the image on top left corner (point 0, 0) of empty cv mat

    // normalize image data
    cv::divide(temp, cv::Scalar(255.0, 255.0, 255.0), temp);          //convert to values from 0-1
    temp -= cv::Scalar(0.485, 0.456, 0.406);                          //subtract the mean from each channel
    cv::divide(temp, cv::Scalar(0.229, 0.224, 0.225), temp);          //divide each channel by standard deviation

    // put the mat inside an std vector
    std::vector<float> img_data;                                                            // declare our std vector
    img_data.assign((float*)temp.data, (float*)temp.data + temp.total()*temp.channels());   // copies the the processed mat to the vector
    return img_data;
}

Observation VisionService::findBinsML(cv::Mat img)
{
    ROS_INFO("Starting machine learning detection for BIN.");
    log(img, 'd');
    // assumes model is already loaded
    // initialize the model's input and output tensors
    auto inpName = new Tensor(model, "input_1");
    auto out_boxes = new Tensor(model, "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3");
    auto out_scores = new Tensor(model, "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3");
    auto out_labels = new Tensor(model, "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3");

    // preprocess image
    float scale;                                                //declare scale variable
    std::vector<float> img_data = preprocess(img, scale, 512);  //process image for input

    // Put data in tensor.
    inpName->set_data(img_data, {1,512,512,3});    //note: change this to 512 + (128*architecture) (eg. d0=512, d1=640, d2=768,etc.)

    // run model
    model.run(inpName, { out_boxes, out_scores, out_labels });

    // convert tensors to std vectors
    auto boxes = out_boxes->get_data<float>();
    auto scores = out_scores->get_data<float>();
    auto labels = out_labels->get_data<int>();

    // scale output boxes back to original image
    for (int i=0; i<boxes.size(); i++)
    {
        boxes[i] = boxes[i] / scale;
    }

    // iterate over results and draw the boxes for visualization!
    for (int i=0; i<scores.size(); i++)
    {
        if (scores[i] > 0.3)
        {
            // extract output values
            float score = scores[i];
            int label = labels[i];
            float xmin = boxes[i*4];    //the boxes come in 4 values: [xmin (left), ymin (top), xmax (right), ymax (bottom)]
            float ymin = boxes[i*4+1];
            float xmax = boxes[i*4+2];
            float ymax = boxes[i*4+3];
            if (label == 0)
                ROS_INFO("Found bin.");
            cv::rectangle(img, {(int)xmin, (int)ymin}, {(int)xmax, (int)ymax}, {154, 209, 8}, 5);    //draws box of detection
            log(img, 'e');                                                                           //logs the image
            return Observation(score, (ymin+ymax)/2, (xmin+xmax)/2, 0);                              //returns observation(score, ycenter, xcenter, 0)
        }
        else
        {                  //since the outputs are already sorted by score
            break;        //if it's lower than the thres then we know it's the last highest one,
        }                 //so we can afford to break out because all the other outputs will be below the threshold
    }
    // if there was nothing over the threshold, then we just return all zeroes :(
    return Observation(0, 0, 0, 0);
}
