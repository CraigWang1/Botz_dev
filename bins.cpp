// Probably don't need to include these things because the sub's script already has it in its header
#include "/content/cppflow/include/Model.h"  //cppflow stuff
#include "/content/cppflow/include/Tensor.h"

#include "opencv2/opencv.hpp"

#include <iostream>

void drawBox(cv::Mat &img, float xmin, float ymin, float xmax, float ymax, std::string classes[], int label, float score, std::vector<cv::Scalar> colors)
{
    /*
    * Helper function to aesthetically draw detected bounding box.
    */
    std::string text = classes[label] + '-' + std::to_string(score);      //setup our label: eg. "bin-0.9996"
    cv::Scalar color = colors[label];                                     //assign color to class
    int baseline = 0;                                                     //baseline variable that the getTextSize function outputs
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);    //get our text size so we can be use it to draw aesthetic text
    cv::rectangle(img, {(int)xmin, (int)ymin}, {(int)xmax, (int)ymax}, color, 3);              //draws detected bbox
    cv::rectangle(img, {(int)xmin, (int)ymax - textSize.height - baseline},                    //draws a highlight behind text for ease of sight
                {(int)xmin + textSize.width, (int)ymax}, color, -1);
    cv::putText(img, text, {(int)xmin, (int)ymax - baseline}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);    //puts text on top of highlight
}

float resize(cv::Mat &image, int image_size)
{
    /*
     * This is a helper function to resize image while preserving aspect ratio;
     * it also returns scale value for rescaling boxes back later.
     */
    int image_height = image.rows;        //initialize image_height variable for convenience
    int image_width = image.cols;         //initialize image_width variable for convenience
    int resized_height, resized_width;    //declare resized_height, resized_width variables to use in the if/else statements
    float scale;                          //scale factor for resizing to original image

    // calculates what height and width to resize to preserve height/width ratios
    if (image_height > image_width) {                     //if the image more tall than wide
        scale = (float)image_size / image_height;     //then the taller side will become the image_size (eg. 512), so the scale is 512/og_h
        resized_height = image_size;                  //downsize taller side to 512
        resized_width = (int)(image_width * scale);   //scale the width down by same scale as the height, preserving side ratios
    }
    else {                                                //otherwise, this means the image is more wide than tall
        scale = (float)image_size / image_width;      //repeat same procedures as above except for width
        resized_height = (int)(image_height * scale);
        resized_width = image_size;
    }
    cv::resize(image, image, cv::Size(resized_width, resized_height));         //resize the image, keeping ratios
    return scale;
}


std::vector<float> preprocess(cv::Mat image, int image_size)
{
    /*
     * This is a helper function that normalizes the image
     * and returns img_data std vector ready for model input (necessary).
     */
    // cv matrix process stuff
    image.convertTo(image, CV_32FC3);                                          // converts to float matrix so we can multiply and divide
    cv::Mat temp(image_size, image_size, CV_32FC3, cv::Scalar(128,128,128));   // makes temporary mat with shape (image_size, image_size, 3) filled with 128s
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
