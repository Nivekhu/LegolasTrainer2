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

	//This is the lbp cascade code
	CascadeClassifier lbp_cascade;
	lbp_cascade.load(fn_lbp);

	//Get a handle on the video device
	VideoCapture cap(deviceId);

	if(!cap.isOpened()){
		cerr << "Capture Device ID: " << deviceId << " cannot be opened." << endl;
		return -1;
	}

	//Current frame of the video device
	Mat frame;
	for(;;){
		cap >> frame;

		//clone the current frame
		Mat original = frame;

		//Convert the current frame to greyscale
		Mat gray;
		cvtColor(frame, gray, COLOR_RGB2GRAY);

		//Creates a vector to store all the faces found
		vector<Rect_<int>> faces;
		lbp_cascade.detectMultiScale(gray, faces);

		Rect face_r; //Rightmost face
		Rect face_l; //Leftmost face

		if(faces.size()) //Makes sure that it only sets the face if there are faces
			face_l = faces[0]; //Sets face_l to the first face since algorithm goes left>right

		//Determines the leftmost and rightmost face		
		for(int i = 0; i < faces.size(); i++){
			Rect face_i = faces[i];
			if(face_l.tl().x > face_i.tl().x)
				face_l = face_i;
			else
				face_r = face_i;
		}

		if(faces.size()){
			if(face_r.tl().x >= face_l.tl().x){
				//Converts both faces to grayscale
				Mat face_rg = gray(face_r);
				Mat face_lg = gray(face_l);

				Mat face_rgr; //Right face Resized
				Mat face_lgr; //Left face Resized

				//Crops the face from the image
				cv::resize(face_rg, face_rgr, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
				cv::resize(face_lg, face_lgr, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

				//Displaying face prediction
				rectangle(original, face_r, CV_RGB(255,0,0),1); //Red outline around right face
				rectangle(original, face_l, CV_RGB(0,255,0),1); //Green outline around left face	

				//Center point for each face
				int posr_x = face_r.tl().x+face_r.width/2;   //Center X position of the Right face
				int posr_y = face_r.tl().y+face_r.height/2;  //Center Y position of the Right face
				int posl_x = face_l.tl().x+face_l.width/2;   //Center X position of the Left face
				int posl_y = face_l.tl().y+face_l.height/2;  //Center Y position of the Left face

				//Display a dot at the center of the faces
				circle(original, Point(posr_x, posr_y), 1.0, CV_RGB(0,0,255), 2.0); //Creates a blue dot on the center of the face
				circle(original, Point(posl_x, posl_y), 1.0, CV_RGB(0,0,255), 2.0); //Creates a blue dot on the center of the face

				//Corners of both faces
				int text_posr_x = std::max(face_r.tl().x - 10, 0);  //Top left x coord of the right Face
				int text_posr_y = std::max(face_r.tl().y - 10, 0);  //Top left y coord of the right Face
				int text_posl_x = std::max(face_l.tl().x - 10, 0);  //Top left x coord of the left Face
				int text_posl_y = std::max(face_l.tl().y - 10, 0);  //Top left y coord of the left Face

				//Text to display info
				string boxtextR = format("x=%d y=%d", posr_x, posr_y); //Right Face info
				string boxtextL = format("x=%d y=%d", posl_x, posl_y); //Left Face info

				//Places the text	
				putText(original, boxtextR, Point(text_posr_x, text_posr_y),FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);
				putText(original, boxtextL, Point(text_posl_x, text_posl_y),FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);
			}
		}

		// Show the result:
		imshow("Legolas Trainer", frame);
		// And display it:
		char key = (char) waitKey(20);
		// Exit this loop on escape:
		if(key == 27)
			break;
	}

	return 0;
}
