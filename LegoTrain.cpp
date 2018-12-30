#include <opencv2/core.hpp>      //This is the core functionality of OpenCV
#include <opencv2/face.hpp>	 //facial recognition functionality
#include <opencv2/highgui.hpp>   //draw on images and output them
#include <opencv2/imgproc.hpp>   //processes images such as converting to grayscale
#include <opencv2/imgcodecs.hpp> //Reading and writing images.
#include <opencv2/objdetect.hpp> //Object regonition algorithms like cascades

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

static void read_csv(const string &filename, vector<Mat> &images, vector<int> &labels, char seperator = ';'){
	std::ifstream file(filename.c_str(), ifstream::in);
	if(!file){
		string error_message = "The filename given seems to be wrong, please doublecheck and try again";
		CV_Error(-5, error_message); //-5 is StsBadArg
	}
	string line,                   //The current line in the csv
	       path,                   //Path to the image
	       classlabel;             //Label at the end of the image
	while(getline(file,line)){   
		stringstream liness(line); 
		getline(liness, path, seperator);
		getline(liness, classlabel);
		if(!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]){
	//This is to set up all the device constants so I don't have to keep typing them in
	string fn_haar = "/home/kevin/opencv/data/haarcascades/haarcascade_frontalface_default.xml";
	string fn_lbp = "/home/kevin/opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml";
	string fn_csv = "/home/kevin/Desktop/LegolasTrainer2/faces.txt";
	int deviceId = 0;

	//Image and label vectors
	vector<Mat> images;
	vector<int> labels;

	// Read in the data (fails if no valid input filename is given, but you'll get an error message):
	try {
		read_csv(fn_csv, images, labels);
	} 
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}

	int im_width = images[0].cols;    //Checks how many columns are in the image matrix
	int im_height = images[0].rows;   //Checks how many rows are in the image matrix

	//Creates the LBPH object
	Ptr<FaceRecognizer> model = LBPHFaceRecognizer::create();

	//Training the face classifier
	cout << "Training..." << endl;
	model->train(images, labels);
	cout << "Done training" << endl;		

}
