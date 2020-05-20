Observation VisionService::findGateML(cv::Mat img)
{
	log(img, 'f');

	//IMPORTANT: Edit architecture of model (eg. efficientdet-d3 -> phi=3)
	int phi = 0;
	int image_sizes[] = {512, 640, 768, 896, 1024, 1152, 1280};      //image sizes that effdet uses
	int image_size = image_sizes[phi];                               //takes the image size that our model uses
	std::string classes[] = {"gate"};                                 //list of classes
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
		if (scores[i] > 0.5) {
			// Extract output values
			float score = scores[i];
			int label = labels[i];
			float xmin = boxes[i*4];    //the boxes come in 4 values: [xmin (left), ymin (top), xmax (right), ymax (bottom)]
			float ymin = boxes[i*4+1];
			float xmax = boxes[i*4+2];
			float ymax = boxes[i*4+3];

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