// Hough transform - na rozpoznavani usecek a dalsich vecich, jako treba kruznic (kruhovych oblouku)
/*
TODO: 1) Podivat se jak detekuji oci a a pokud nedetekuji kazde vzlast, tak udelat trenovani.
			- pozitivni / negativni vzorky
	  2) Rozjet detekci vicek Houghovou trans. nebo prijit s jinym konceptem
	  3) Rohovka a duhovka - asi zase Houghova trans., takze by se to dalo vzit s vickem

	  // mrl.cs.vsb.cz/eyes/dataset/video
*/

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
void irisDetection( Mat eye, string windowName, int x, int y, int frameX, int frameY);
void betterIrisDetection( Mat eye, string windowName, int x, int y, int frameX, int frameY);
void irisAndPupilDetection ( Mat eye, string windowName, int x, int y, int frameX, int frameY );
void pupilDetection( Mat src );
void myPupilDetection( Mat eye, string windowName, int x, int y, int frameX, int frameY);

void showWindowAtPosition( string imageName, Mat mat, int x, int y );
bool isContainigNumber(int array[], int size, int number);
vector<Rect> pickEyes(vector<Rect> eyes, Mat face);
Rect pickFace(vector<Rect> faces);

//trackbars
void onHCParame2Trackbar(int pos, void *);
void onHCDpTrackbar(int pos, void *);
void onHCMinDistanceTrackbar(int pos, void *);



// default values
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

string file = "lena.png";
bool useVideo = false, useCamera = false, stepFrame = false, showWindow = false;
Mat frame, originalFrame;

// sliders
const int sliderHCParam2max = 300;
int sliderHCParam2, HCParam2;

const int sliderHCDpMax = 200;
int sliderHCDp;
double HCDp;

const int sliderHCMinDistanceMax = 200;
int sliderHCMinDistance;
double HCMinDistance;


int main( int argc, const char** argv )
{
	processArguments( argc, argv);

	CvCapture* capture;
	// Mat frame;

	sliderHCParam2 = HCParam2 = 40;		//38
	sliderHCDp = 20;	// deli se to 10...
	HCDp = 2;
	sliderHCMinDistance = HCMinDistance = 100;	// toto se odviji dost podle velikosti obrazku... (?)

   	loadCascades();

	if (useVideo)
	{
		capture = cvCaptureFromFile(file.c_str());

		//cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, 60);	// posun na 60 frame
	}
	else if (useCamera)
	{
		capture =  cvCaptureFromCAM( -1 );
	}
	else
	{
		frame = imread(file);
		originalFrame = frame.clone();
		detectAndDisplay(frame);
		//irisDetection(frame, "iris");
		
		waitKey(0);
	}
	
	if (useVideo || useCamera)
	{
		if( capture )
		{
			while( true )
			{
				if (stepFrame)
				{
					int c = waitKey(10);
					
					if( (char)c == 'n' || (char)c == 'N' || showWindow)
					{
						frame = cvQueryFrame( capture );
						originalFrame = frame.clone();
				
						//-- 3. Apply the classifier to the frame
						if( !frame.empty() )
						{
							detectAndDisplay( frame );					
						}
						else
						{
							printf(" --(!) No captured frame -- Break!"); break;
						}					

						c = -1;
						showWindow = false;
					}
				}
				// normalni stav
				else
				{
					frame = cvQueryFrame( capture );
					originalFrame = frame.clone();
				
					//-- 3. Apply the classifier to the frame
					if( !frame.empty() )
					{
						detectAndDisplay( frame );					
					}
					else
					{
						printf(" --(!) No captured frame -- Break!"); break;
					}					
				}	

				int c = waitKey(10);
				if( (char)c == 'c' || (char)c == 'C' )
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
        else if ((arg == "-s") || (arg == "--step"))
        {
        	stepFrame = true;
        	showWindow = true;
        }
        else if ((arg == "-c") || (arg == "--camera"))
        {
        	useCamera = true;
        }
	}

	return 0;
 }

void showUsage( string name )
{
    cerr << "Usage: " << name << " <option(s)> SOURCES"
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
	vector<Rect> faces;
  	Mat frame_gray;

	// convert from color to grayscale
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	
	// contrast adjustment using the image's histogram
  	equalizeHist( frame_gray, frame_gray );
	
  	//Detects objects (faces) of different sizes in the input image. The detected objects are returned as a list of rectangles.
  	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(frame_gray.size().width / 6, frame_gray.size().height / 6) );

  	if (faces.size() > 0)
  	{
  		// zavolat f-ci na spravny face. Nemusi to byt ve foru pak taaaaada
	  	Rect face = pickFace(faces);

    	Point center( face.x + face.width*0.5, face.y + face.height*0.5 );
    	rectangle( frame, Rect(face.x, face.y, face.width, face.height), Scalar( 255, 0, 255 ), 4, 8, 0 );

    	Mat faceROI = frame_gray( face );
    	std::vector<Rect> eyes;

    	//-- In each face, detect eyes
    	eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );


    	eyes = pickEyes(eyes, faceROI);

    	for( size_t j = 0; j < eyes.size(); j++ )
     	{
     		rectangle( frame, Rect(face.x + eyes[j].x, face.y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar( 0, 0, 255 ), 4, 8, 0 );

       		Mat eyeMat = faceROI(Rect(eyes[j].x, eyes[j].y, eyes[j].width, eyes[j].height));

			char numstr[21]; // enough to hold all numbers up to 64-bits
			sprintf(numstr, "%d", static_cast<int>(j + 1));
			string eyeName = "eye";
			
			betterIrisDetection(eyeMat, eyeName + numstr, 520 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
     	}	
    }
  	
  	imshow( window_name, frame );
  	createTrackbar("HC param2", window_name, &sliderHCParam2, sliderHCParam2max, onHCParame2Trackbar);
  	createTrackbar("HC dp", window_name, &sliderHCDp, sliderHCDpMax, onHCDpTrackbar);
  	createTrackbar("HC min distance", window_name, &sliderHCMinDistance, sliderHCMinDistanceMax, onHCMinDistanceTrackbar);
}


Rect pickFace(vector<Rect> faces)
{

	// vrati nejvetsi oblicej
	double max = 0;
	int maxIndex = -1;

	for (int i = 0; i < faces.size(); ++i)
	{
		int volume = faces[i].size().width * faces[i].size().height;
		if (volume > max)
		{
			max = volume;
			maxIndex = i;
		}
	}

	if (maxIndex >= 0)
	{
		return faces[maxIndex];
	}
	else
	{
		return faces[0];
	}
}

// Podivat se/popremyslet jak vyfiltrovat false detected oci
vector<Rect> pickEyes(vector<Rect> eyes, Mat face)
{
	vector<Rect> correctEyes = eyes;

	// prostor pro oci je urcite ve vrchni polovine obliceje...
	for (int i = 0; i < eyes.size(); ++i)
	{
		if (eyes[i].y > (face.size().height * 0.5 ))
		{
			correctEyes.erase(correctEyes.begin() + i);
			cout << "Mazu! Oblast oka mimo vrchni polovinu obliceje. x,y = " << eyes[i].x << ", " << eyes[i].y << ". Polovina obliceje ma delku " << face.size().height * 0.5 << endl;
		}
	}

	// odebere ocni oblasti, ktere zasahuji mimo oblicej
	for (int i = 0; i < correctEyes.size(); ++i)
	{
		// Prave oko
		if ( eyes[i].x > (face.size().width * 0.5) )
		{
			if ( (eyes[i].x + eyes[i].width)  > face.size().width )
			{
				cout << "Mazu! Oblast praveho oka je mimo oblicej. x,y = " << eyes[i].x << ", " << eyes[i].y << endl;
				correctEyes.erase(correctEyes.begin() + i);			
			}
		}
		// Leve oko
		else
		{
			if ( eyes[i].x < 0 || (eyes[i].x + eyes[i].width)  > (face.size().width * 0.5 ) )
			{
				cout << "Mazu! Oblast leveho oka je mimo oblicej. x,y = " << eyes[i].x << ", " << eyes[i].y << endl;
				correctEyes.erase(correctEyes.begin() + i);			
			}
		}
	}

	// odstrani oci s podobnym stredem
	for (int i = 0; i < correctEyes.size(); ++i)
	{
		// jak jsou vzdalene stredy 2. ocnich oblasti. Pokud je to min nez treshold (relativne), tak mensi ocni oblast odstranime
		double distancesTresh = 0.1;	

		for (int j = 0; j < correctEyes.size(); ++j)
		{
			if (i != j)
			{
				double distance = sqrt( pow(correctEyes[i].x - correctEyes[j].x, 2) + pow(correctEyes[i].y - correctEyes[j].y, 2) );

				if (face.size().width != 0)
				{
					//cout << "distance = " << distance / face.size().width << endl;
					if (distance / face.size().width < distancesTresh)
					{
						// smaze mensi ze 2 ocnich oblasti
						if (correctEyes[i].width > correctEyes[j].width)
						{
							correctEyes.erase(correctEyes.begin() + j);
						}
						else
						{
							correctEyes.erase(correctEyes.begin() + i);
						}
					}
				}
			}
		}
	}

	

	if (correctEyes.size() > 0)
	{
		return correctEyes;
	}
	else
	{
		return eyes;
	}
}

void refreshImage()
{
	frame = originalFrame.clone();
	detectAndDisplay(frame);
}

void onHCParame2Trackbar(int pos, void *)
{
	HCParam2 = pos;

	cout << "HC param2 = " << HCParam2 << endl;

	refreshImage();
}

void onHCDpTrackbar(int pos, void *)
{
	HCDp = pos / 10.;

	if (HCDp <= 0)
		HCDp = 1;

	cout << "HC dp = " << HCDp << endl;

	refreshImage();
}

void onHCMinDistanceTrackbar(int pos, void *)
{
	HCMinDistance = pos;

	if (HCMinDistance <= 0)
		HCMinDistance = 1;

	cout << "HC Min Distance = " << HCMinDistance << endl;

	refreshImage();
}

void betterIrisDetection ( Mat eye, string windowName, int x, int y, int frameX, int frameY )
{
	vector<Vec3f> circles;
	
	Mat gaussienEye;
	GaussianBlur( eye, gaussienEye, Size(5, 5), 2, 2 );	//bilateralFilter ???

	//Mat sobel;
	//Sobel(gaussienEye, sobel, gaussienEye.depth(), 1, 0, 3);
	// showWindowAtPosition( windowName, sobel, x, y );


	Mat uselessMat;
	double otsu_thresh_val = threshold( eye, uselessMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
	double high_thresh_val  = otsu_thresh_val;
	double lower_thresh_val = otsu_thresh_val * 0.5;
	//cout << "Computed tresholds = " << high_thresh_val << ", " << lower_thresh_val << endl;	//140,70

	// polomery
	int minRadius = cvRound(gaussienEye.size().width * 0.1);	
	int maxRadius = cvRound(gaussienEye.size().width * 0.3);	//0.3
	HoughCircles( gaussienEye, circles, CV_HOUGH_GRADIENT, HCDp, HCMinDistance, high_thresh_val, HCParam2, minRadius, maxRadius);	//eyeCanny.rows / 8
	
	cvtColor(gaussienEye, gaussienEye, CV_GRAY2BGR);
	/// Draw the circles detected
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		// circle outline
		circle( gaussienEye, center, radius, Scalar(0, 0, 255), 0, 8, 0 );
		
		Point frameCenter(cvRound(circles[i][0]) + frameX, cvRound(circles[i][1]) + frameY);
		// circle outline
		circle( frame, frameCenter, radius, Scalar(0,0,255), 3, 8, 0 );	
	}

	showWindowAtPosition( windowName, gaussienEye, x, y );
}

void irisDetection( Mat eye, string windowName, int x, int y, int frameX, int frameY)
{
	vector<Vec3f> circles;

	//cvSmooth( eye, eye, CV_GAUSSIAN, 9, 9 );
	//cvtColor( eye, eye, CV_BGR2GRAY );
	
	Mat gaussienEye;
	GaussianBlur( eye, gaussienEye, Size(5, 5), 2, 2 );	//bilateralFilter ???


	Mat uselessMat;
	double otsu_thresh_val = threshold( eye, uselessMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
	double high_thresh_val  = otsu_thresh_val;
	double lower_thresh_val = otsu_thresh_val * 0.5;
	cout << "Computed tresholds = " << high_thresh_val << ", " << lower_thresh_val << endl;

	Mat eyeCanny;
	Canny(gaussienEye, eyeCanny, lower_thresh_val, high_thresh_val);
	//eyeCanny = gaussienEye;

	//showWindowAtPosition( windowName, eyeCanny, x, y );

	int minRadius = cvRound(eyeCanny.size().width * 0.1);
	int maxRadius = cvRound(eyeCanny.size().width * 0.6);	//0.4
	
	/// Apply the Hough Transform to find the circles
	HoughCircles( eyeCanny, circles, CV_HOUGH_GRADIENT, 2, eyeCanny.rows / 2, high_thresh_val, HCParam2, minRadius, maxRadius);
	cout << "num of circles = " << circles.size() << endl;

	/// Draw the circles detected
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		//circle( eye, center, 3, Scalar(0,255,0), -1, 8, 0 );
		// circle outline
		//circle( eye, center, radius, Scalar(0,0,255), 3, 8, 0 );
		
		Point frameCenter(cvRound(circles[i][0]) + frameX, cvRound(circles[i][1]) + frameY);
		// circle center
		//circle( frame, frameCenter, 3, Scalar(0,255,0), -1, 8, 0 );
		// circle outline
		circle( frame, frameCenter, radius, Scalar(0,0,255), 3, 8, 0 );	
	}

	showWindowAtPosition( windowName, eyeCanny, x, y );
}



void irisAndPupilDetection ( Mat eye, string windowName, int x, int y, int frameX, int frameY )
{
	vector<Vec3f> circles;
	
	//cvSmooth( eye, eye, CV_GAUSSIAN, 9, 9 );
	//cvtColor( eye, eye, CV_BGR2GRAY );

	Mat gaussienEye;
	GaussianBlur( eye, gaussienEye, Size(5, 5), 2, 2 );	//bilateralFilter ???

	Mat uselessMat;
	double otsu_thresh_val = threshold( eye, uselessMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
	double high_thresh_val  = otsu_thresh_val;
	double lower_thresh_val = otsu_thresh_val * 0.5;
	//cout << "Computed tresholds = " << high_thresh_val << ", " << lower_thresh_val << endl;	//140,70

	Mat eyeCanny;
	Canny(gaussienEye, eyeCanny, lower_thresh_val, high_thresh_val);
	//Canny(gaussienEye, eyeCanny, 50, 120);
	//Canny(gaussienEye, eyeCanny, cannyLow, cannyHigh);
	//showWindowAtPosition( windowName, eyeCanny, x, y );

	int minRadius = cvRound(eyeCanny.size().width * 0.1);
	int maxRadius = cvRound(eyeCanny.size().width * 0.3);
	HoughCircles( eyeCanny, circles, CV_HOUGH_GRADIENT, 2, eyeCanny.rows / 8, high_thresh_val, HCParam2, minRadius, maxRadius);
	//cout << "num of circles = " << circles.size() << endl;

	vector<Vec3f> pupils;
	HoughCircles( eyeCanny, pupils, CV_HOUGH_GRADIENT, 2, 1, 100, 300, 0, 0);
	//cout << "num of small circles = " << pupils.size() << endl;

	// hleda kruznice se stejnym stredem

	vector<Vec3f> irisesAndPupils;
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center1(cvRound(circles[i][0]), cvRound(circles[i][1]));

		for (int j = 0; j < pupils.size(); ++j)
		{
			Point center2(cvRound(pupils[j][0]), cvRound(pupils[j][1]));


			double distance = sqrt(pow(circles[i][0] - pupils[j][0], 2) - pow(circles[i][1] - pupils[j][1], 2));
			//cout << distance << endl;
			if (center1 == center2)
			{
				cout << "stejne stredy maji kruznice " << i << " a " << j << endl;
				irisesAndPupils.push_back(circles[i]);
				irisesAndPupils.push_back(pupils[j]);
			}
		}
	}


	cvtColor(eyeCanny, eyeCanny, CV_GRAY2BGR);
	/// Draw the circles detected
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		//circle( eye, center, 3, Scalar(0,255,0), -1, 8, 0 );
		// circle outline
		circle( eyeCanny, center, radius, Scalar(0, 0, 255), 0, 8, 0 );
		
		Point frameCenter(cvRound(circles[i][0]) + frameX, cvRound(circles[i][1]) + frameY);
		// circle center
		//circle( frame, frameCenter, radius, Scalar(0,255,0), 3, 8, 0 );
		// circle outline
		circle( frame, frameCenter, radius, Scalar(0,0,255), 3, 8, 0 );	
	}

	for( size_t i = 0; i < pupils.size(); i++ )
	{
		Point center(cvRound(pupils[i][0]), cvRound(pupils[i][1]));
		int radius = cvRound(pupils[i][2]);

		//circle( eye, center, 3, Scalar(0,255,0), -1, 8, 0 );
		// circle outline
		circle( eyeCanny, center, radius, Scalar(0, 255, 255), 0, 8, 0 );
		
		Point frameCenter(cvRound(pupils[i][0]) + frameX, cvRound(pupils[i][1]) + frameY);
		// circle center
		//circle( frame, frameCenter, radius, Scalar(0,255,0), 3, 8, 0 );
		// circle outline
		circle( frame, frameCenter, radius, Scalar(0,255,255), 3, 8, 0 );	
	}

	/*for( size_t i = 0; i < irisesAndPupils.size(); i++ )
	{
		Point center(cvRound(irisesAndPupils[i][0]), cvRound(irisesAndPupils[i][1]));
		int radius = cvRound(irisesAndPupils[i][2]);

		//circle( eye, center, 3, Scalar(0,255,0), -1, 8, 0 );
		// circle outline
		circle( eyeCanny, center, radius, Scalar(0, 0, 255), 0, 8, 0 );
		
		Point frameCenter(cvRound(irisesAndPupils[i][0]) + frameX, cvRound(irisesAndPupils[i][1]) + frameY);
		// circle center
		//circle( frame, frameCenter, radius, Scalar(0,255,0), 3, 8, 0 );
		// circle outline
		circle( frame, frameCenter, radius, Scalar(0,0,255), 1, 8, 0 );	
	}*/

	showWindowAtPosition( windowName, eyeCanny, x, y );
}

bool isContainigNumber(int array[], int size, int number)
{
	for (int i = 0; i < size; ++i)
	{
		if (array[i] == number)
			return true;
	}

	return false;
}

void myPupilDetection( Mat eye, string windowName, int x, int y, int frameX, int frameY)
{
	vector<Vec3f> circles;
	

	/*Mat gaussienEye;
	GaussianBlur( eye, gaussienEye, Size(5, 5), 2, 2 );	//bilateralFilter ???

	//Mat sobel;
	//Sobel(gaussienEye, sobel, gaussienEye.depth(), 1, 0, 3);
	// showWindowAtPosition( windowName, sobel, x, y );


	Mat uselessMat;
	double otsu_thresh_val = threshold( eye, uselessMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
	double high_thresh_val  = otsu_thresh_val;
	double lower_thresh_val = otsu_thresh_val * 0.5;
	cout << "Computed tresholds = " << high_thresh_val << ", " << lower_thresh_val << endl;	//140,70

	Mat eyeCanny;
	Canny(gaussienEye, eyeCanny, lower_thresh_val, high_thresh_val);
	//Canny(gaussienEye, eyeCanny, 50, 120);
	//Canny(gaussienEye, eyeCanny, cannyLow, cannyHigh);
	//showWindowAtPosition( windowName, eyeCanny, x, y );

	int minRadius = cvRound(eyeCanny.size().width * 0.1);
	int maxRadius = cvRound(eyeCanny.size().width * 0.3);
	HoughCircles( eyeCanny, circles, CV_HOUGH_GRADIENT, 2, eyeCanny.rows / 4, high_thresh_val, param2HoughC, minRadius, maxRadius);
	cout << "num of circles = " << circles.size();

	//vector<Vec3f> irises = getCorrectIrises(circles);

	cvtColor(eyeCanny, eyeCanny, CV_GRAY2BGR);
	/// Draw the circles detected
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		//circle( eye, center, 3, Scalar(0,255,0), -1, 8, 0 );
		// circle outline
		circle( eyeCanny, center, radius, Scalar(0, 0, 255), 0, 8, 0 );
		
		Point frameCenter(cvRound(circles[i][0]) + frameX, cvRound(circles[i][1]) + frameY);
		// circle center
		//circle( frame, frameCenter, radius, Scalar(0,255,0), 3, 8, 0 );
		// circle outline
		circle( frame, frameCenter, radius, Scalar(0,0,255), 3, 8, 0 );	
	}

	showWindowAtPosition( windowName, uselessMat, x, y );*/
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