// Hough transform - na rozpoznavani usecek a dalsich vecich, jako treba kruznic (kruhovych oblouku)
/*
TODO: 1) Podivat se jak detekuji oci a a pokud nedetekuji kazde vzlast, tak udelat trenovani.
			- pozitivni / negativni vzorky
	  2) Rozjet detekci vicek Houghovou trans. nebo prijit s jinym konceptem
	  3) Rohovka a duhovka - asi zase Houghova trans., takze by se to dalo vzit s vickem

	  // mrl.cs.vsb.cz/eyes/dataset/video
*/
\
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


 // Function Headers
int processArguments( int argc, const char** argv );
void showUsage( string name );

void loadCascades();
void detectAndDisplay( Mat frame );
void pupilDetection( Mat src );
void myPupilDetection( Mat eye, string windowName, int x, int y );

void showWindowAtPosition( string imageName, Mat mat, int x, int y );


// default values
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

string file = "Lenna.png";
bool useVideo = false;


int main( int argc, const char** argv )
{
	processArguments( argc, argv);

	CvCapture* capture;
	Mat frame;

   	loadCascades();

	//-- 2. Read the video stream	
	if (useVideo)
	{
		//capture =  cvCaptureFromCAM( -1 );
		capture = cvCaptureFromFile(file.c_str());

		//cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, 60);	// posun na 60 frame
	}
	else
	{
		frame = imread(file);
		detectAndDisplay(frame);
		
		waitKey(0);
	}
	
	
	
	if (useVideo)
	{
		if( capture )
		{
			while( true )
			{
				frame = cvQueryFrame( capture );

				//-- 3. Apply the classifier to the frame
				if( !frame.empty() )
				{
					detectAndDisplay( frame );					
				}
				else
				{
					printf(" --(!) No captured frame -- Break!"); break;
				}

				int c = waitKey(10);
				if( (char)c == 'c' )
				{
					break;
				}
			}
		}
}
	
	//pupilDetection();
	
	return 0;
 }

 int processArguments( int argc, const char** argv )
 {
 	cout << argc << endl;
	for (int i = 1; i < argc; ++i)
	{	 	
		string arg = argv[i];
        if ((arg == "-h") || (arg == "--help"))
        {
            showUsage(argv[0]);
            return 0;
        }
        else if ((arg == "-f") || (arg == "--file"))
        {
        	// Make sure we aren't at the end of argv!
        	if (i + 1 < argc)
        	{ 
                file = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
            } 
            // Uh-oh, there was no argument to the destination option.
            else 
            { 
            	std::cerr << "--file option requires one argument." << std::endl;
            	return 1;
            }
        }
        else if ((arg == "-v") || (arg == "--video"))
        {
        	useVideo = true;
        }
	}

	return 0;
 }

void showUsage( string name )
{
    std::cerr << "Usage: " << name << " <option(s)> SOURCES"
              << "Options:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t-d,--destination DESTINATION\tSpecify the destination path"
              << std::endl;
}

void loadCascades()
{
	//-- 1. Load the cascades
   	if( !face_cascade.load( face_cascade_name ) )
	{ 
		printf("--(!)Error loading\n");
		//return -1; 
	}
	
   	if( !eyes_cascade.load( eyes_cascade_name ) )
	{ 
		printf("--(!)Error loading\n");
		//return -1; 
	}
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
  	Mat frame_gray;

	// convert from color to grayscale
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	//imshow( "gray image", frame_gray );
	
	// contrast adjustment using the image's histogram
  	equalizeHist( frame_gray, frame_gray );
	//imshow( "gray equalize image", frame_gray );

  	//Detects objects (faces) of different sizes in the input image. The detected objects are returned as a list of rectangles.
  	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(frame_gray.size().width / 4, frame_gray.size().height / 5) );

  	for( size_t i = 0; i < faces.size(); i++ )
  	{
    	Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    	ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    	Mat faceROI = frame_gray( faces[i] );
    	std::vector<Rect> eyes;

    	//-- In each face, detect eyes
    	eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    	for( size_t j = 0; j < eyes.size(); j++ )
     	{
       		Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       		int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       		//circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
       		
       		float rectWidth = (eyes[j].width + eyes[j].height)*0.5;
       		//rectangle( frame, Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, rectWidth, rectWidth), Scalar( 0, 0, 255 ), 4, 8, 0 );

       		Mat eyeMat = frame(Rect(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y, rectWidth, rectWidth));

			char numstr[21]; // enough to hold all numbers up to 64-bits
			sprintf(numstr, "%d", static_cast<int>(j + 1));
			string eyeName = "eye";
			
			//showWindowAtPosition( eyeName + numstr, eyeMat, 200 + 80 * j, 200 );

			//pupilDetection(eyeMat);
			myPupilDetection(eyeMat, eyeName + numstr, 200 + 80 * j, 200 );
     	}	

     	if (eyes.size() <= 0)
     	{
     		cout << "********* MRK *********" << endl;
     	}
		
  	}
  	//-- Show what you got
  	//imshow( window_name, frame );
}

void myPupilDetection( Mat eye, string windowName, int x, int y )
{
	Mat eye_gray;

	


	 showWindowAtPosition( windowName, eye, x, y );
}

void pupilDetection( Mat src )
{
// Load image
	//cv::Mat src = cv::imread("ja.jpg");
	// if (src.empty())
// 		return -1;

	// Invert the source image and convert to grayscale
	cv::Mat gray;
	cv::cvtColor(~src, gray, CV_BGR2GRAY);
	//cv::imshow("image1", gray);

	// Convert to binary image by thresholding it
	cv::threshold(gray, gray, 220, 255, cv::THRESH_BINARY);
	//cv::imshow("image2", gray);

	// Find all contours
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	// Fill holes in each contour
	cv::drawContours(gray, contours, -1, CV_RGB(255,255,255), -1);

	for (int i = 0; i < contours.size(); i++)
	{
		double area = cv::contourArea(contours[i]);
		cv::Rect rect = cv::boundingRect(contours[i]);
		int radius = rect.width/2;

		// If contour is big enough and has round shape
		// Then it is the pupil
		if (area >= 30 //&& 
		    //std::abs(1 - ((double)rect.width / (double)rect.height)) <= 0.2 &&
				//std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2
			)	
		{
			cv::circle(src, cv::Point(rect.x + radius, rect.y + radius), radius, CV_RGB(255,0,0), 2);
		}
	}

	cv::imshow("final image", src);
	//cv::waitKey(0);
}



void showWindowAtPosition( string imageName, Mat mat, int x, int y )
{
	imshow( imageName, mat );
	moveWindow(imageName, x, y);
}