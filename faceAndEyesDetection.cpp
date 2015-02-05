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
#include "opencv2/photo/photo.hpp"


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
void betterIrisDetection ( Mat eye, string windowName, int x, int y, int frameX, int frameY, vector<Vec6f> *eyes);
void pupilCenterDetection ( Mat eye, string windowName, int x, int y, int frameX, int frameY);
void irisAndPupilDetection ( Mat eye, string windowName, int x, int y, int frameX, int frameY );
void pupilDetection( Mat src );
void myPupilDetection( Mat eye, string windowName, int x, int y, int frameX, int frameY);

void showWindowAtPosition( string imageName, Mat mat, int x, int y );
void refreshImage();
bool isContainigNumber(int array[], int size, int number);
vector<Rect> pickEyeRegions(vector<Rect> eyes, Mat face);
Rect pickFace(vector<Rect> faces);

void pickCorrectIrises();

void drawIrises();
void drawEyesCentres();

Rect setEyesCentres ( Mat eye, string windowName, int x, int y, int frameX, int frameY);
void myCircleHough(Mat eye, string windowName, int x, int y, int frameX, int frameY, Point center);
Mat removeReflections(Mat eye, string windowName, int x, int y, int frameX, int frameY);


//trackbars
void onHCParam1Trackbar(int pos, void *);
void onHCParam2Trackbar(int pos, void *);
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
bool drawInFrame = true;
Mat frame, originalFrame;

// sliders
const int sliderHCParam1max = 300;
int sliderHCParam1, HCParam1;

const int sliderHCParam2max = 300;
int sliderHCParam2, HCParam2;

const int sliderHCDpMax = 200;	// deli se 10
int sliderHCDp;
double HCDp;

const int sliderHCMinDistanceMax = 200;
int sliderHCMinDistance, HCMinDistance;

vector<Point> eyesCentres;
vector<Vec6f> allEyes;
vector<Vec6f> eye1;
vector<Vec6f> eye2;
vector<Vec3f> irises;

int main( int argc, const char** argv )
{
	processArguments( argc, argv);

	CvCapture* capture;
	// Mat frame;

	sliderHCParam1 = HCParam1 = 180;	//26	//35
	sliderHCParam2 = HCParam2 = 19;		//21	//30
	
	sliderHCDp = 17;	// deli se to 10...	// 30
	HCDp = 1.7;	// 3
	
	sliderHCMinDistance = HCMinDistance = 1;	// 170	//57

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
					else if( (char)c == 'i' || (char)c == 'I' || showWindow)
					{
						cout << "HC param 1 = " << HCParam1 <<  "HC param 2 = " << HCParam2 << ", HC dp = " << HCDp << ", HC min distance = " << HCMinDistance << endl;
					}
					else if( (char)c == 'f' || (char)c == 'F' || showWindow)
					{
						drawInFrame = !drawInFrame;

						refreshImage();
					}
					// else if( (char)c == 'p' || (char)c == 'P' || showWindow)
					// {
					// 	stepFrame = false;
					// }
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
				else if( (char)c == 'p' || (char)c == 'P' || showWindow)
				{
					stepFrame = !stepFrame;
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
  	allEyes.clear();
  	eye1.clear();
  	eye2.clear();
  	eyesCentres.clear();
  	irises.clear();


	// convert from color to grayscale
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	
	// contrast adjustment using the image's histogram
  	equalizeHist( frame_gray, frame_gray );
 	
 	//  Ptr<CLAHE> clahe = createCLAHE();
	// clahe->setClipLimit(3);
	// clahe->apply(frame_gray,frame_gray);
	
  	//Detects objects (faces) of different sizes in the input image. The detected objects are returned as a list of rectangles.
  	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(frame_gray.size().width / 6, frame_gray.size().height / 6) );

  	if (faces.size() > 0)
  	{
  		Rect face = pickFace(faces);

    	Point center( face.x + face.width*0.5, face.y + face.height*0.5 );
    	rectangle( frame, Rect(face.x, face.y, face.width, face.height), Scalar( 255, 0, 255 ), 4, 8, 0 );

    	Mat faceROI = frame_gray( face );
    	std::vector<Rect> eyes;

    	//-- In each face, detect eyes
    	eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );


    	eyes = pickEyeRegions(eyes, faceROI);

    	for( size_t j = 0; j < eyes.size(); j++ )
     	{
     		rectangle( frame, Rect(face.x + eyes[j].x, face.y + eyes[j].y, eyes[j].width, eyes[j].height), Scalar( 0, 0, 255 ), 4, 8, 0 );

       		Mat eyeMat = faceROI(Rect(eyes[j].x, eyes[j].y, eyes[j].width, eyes[j].height));

			char numstr[21]; // enough to hold all numbers up to 64-bits
			sprintf(numstr, "%d", static_cast<int>(j + 1));
			string eyeName = "eye";

			eyeMat = removeReflections(eyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);

			Rect smallEyeRect = setEyesCentres(eyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);

			Mat smallEyeMat = eyeMat(smallEyeRect);
			//Prozatimni hack na pozici ve framu
			int tmpX = eyeMat.size().width - smallEyeMat.size().width;
			int tmpY = eyeMat.size().height - smallEyeMat.size().height;

			/*if (j == 0)
			{
				betterIrisDetection(smallEyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x + tmpX*0.5, face.y + eyes[j].y + tmpY*0.5, &eye1);
			}
			else
			{
				betterIrisDetection(smallEyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x + tmpX*0.5, face.y + eyes[j].y + tmpY*0.5, &eye2);
			}	*/		
			
			//pupilCenterDetection(eyeMat, eyeName + numstr, 520 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);

			myCircleHough(eyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, Point(smallEyeRect.x + smallEyeRect.width*0.5, smallEyeRect.y + smallEyeRect.height*0.5));
     	}	

     	pickCorrectIrises();

		if (drawInFrame)
		{
	     	//drawIrises();
	     	drawEyesCentres();
	    }
    }


  	
  	imshow( window_name, frame );

  	createTrackbar("HC param1", window_name, &sliderHCParam1, sliderHCParam1max, onHCParam1Trackbar);
  	createTrackbar("HC param2", window_name, &sliderHCParam2, sliderHCParam2max, onHCParam2Trackbar);
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

vector<Rect> pickEyeRegions(vector<Rect> eyes, Mat face)
{
	vector<Rect> correctEyes = eyes;

	// prostor pro oci je urcite ve vrchni polovine obliceje...  !!! toto by se dalo udelat i zmensenim oblasti obliceje o 1/2 -> lepsi vykon!!!
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
				double distance = sqrt( pow(correctEyes[i].x - correctEyes[j].x, 2.) + pow(correctEyes[i].y - correctEyes[j].y, 2.) );

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

	// TMP - tvrde smazeni na 2 ocni oblasti
	if (correctEyes.size() > 2)
		correctEyes.erase(correctEyes.begin() + 2, correctEyes.begin() + correctEyes.size());  

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
	//if (stepFrame)
	{
		frame = originalFrame.clone();
		detectAndDisplay(frame);
	}
}

void onHCParam1Trackbar(int pos, void *)
{
	HCParam1 = pos;

	cout << "HC param1 = " << HCParam1 << endl;

	if (HCParam1 < 1)
		HCParam1 = 1;

	refreshImage();
}

void onHCParam2Trackbar(int pos, void *)
{
	HCParam2 = pos;

	cout << "HC param2 = " << HCParam2 << endl;

	if (HCParam2 < 1)
		HCParam2 = 1;

	refreshImage();
}

void onHCDpTrackbar(int pos, void *)
{
	HCDp = pos / 10.;

	if (HCDp < 1)
		HCDp = 1;

	cout << "HC dp = " << HCDp << endl;

	refreshImage();
}

void onHCMinDistanceTrackbar(int pos, void *)
{
	HCMinDistance = pos;

	if (HCMinDistance < 1)
		HCMinDistance = 1;

	cout << "HC Min Distance = " << HCMinDistance << endl;

	refreshImage();
}

void betterIrisDetection ( Mat eye, string windowName, int x, int y, int frameX, int frameY, vector<Vec6f> *eyes)
{
	//showWindowAtPosition( windowName + "_normalized", eye, x, y );

	vector<Vec3f> circles;
	
	Mat gaussienEye;
	GaussianBlur( eye, gaussienEye, Size(5, 5), 2, 2 );	//bilateralFilter ???

	/*
	// **** Sobel *****
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, grad;
	
	int scale = 1;
  	int delta = 0;
  	int ddepth = CV_16S;

	/// Gradient X
	Sobel( gaussienEye, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( gaussienEye, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );

	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	showWindowAtPosition( windowName + "sobel", grad, x, y );

	// **** END Sobel *****
	*/


	Mat uselessMat;
	double otsu_thresh_val = threshold( gaussienEye, uselessMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU );
	double high_thresh_val  = otsu_thresh_val;
	double lower_thresh_val = otsu_thresh_val * 0.5;
	//cout << "Computed tresholds = " << high_thresh_val << ", " << lower_thresh_val << endl;	//140,70

	// Mat eyeCanny;
	//Canny(gaussienEye, gaussienEye, lower_thresh_val, high_thresh_val);
	//showWindowAtPosition( windowName + "_canny", gaussienEye, x, y );

	// polomery
	int minRadius = cvRound(gaussienEye.size().width * 0.2);	
	int maxRadius = cvRound(gaussienEye.size().width * 0.45);	//0.3
	HoughCircles( gaussienEye, circles, CV_HOUGH_GRADIENT, HCDp, HCMinDistance, high_thresh_val, HCParam2, minRadius, maxRadius);	//eyeCanny.rows / 8, high_thresh_val
	

	/// Add cicles to allEyes
	for( size_t i = 0; i < circles.size(); i++ )
	{
		float relativeXPosition = circles[i][0] / eye.size().width;
		float relativeYPosition = circles[i][1] / eye.size().height;
		//cout << "relativePosition: " << circles[i][0] << " / " << eye.size().width << " = " << relativeXPosition << endl;

		eyes->push_back(Vec6f(circles[i][0] + frameX, circles[i][1] + frameY, circles[i][2], relativeXPosition , relativeYPosition, 1));	//x, y, radius, relativni pozice, "barva", nada
	}

	cvtColor(gaussienEye, gaussienEye, CV_GRAY2BGR);
	/// Draw the circles detected
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		// circle outline
		circle( gaussienEye, center, radius, Scalar(0, 0, 255), 0, 8, 0 );
		
		//Point frameCenter(cvRound(circles[i][0]) + frameX, cvRound(circles[i][1]) + frameY);
		// circle outline
		//circle( frame, frameCenter, radius, Scalar(0,0,255), 3, 8, 0 );	
	}

	 showWindowAtPosition( windowName, gaussienEye, x, y + 260);	
}

Rect setEyesCentres ( Mat eye, string windowName, int x, int y, int frameX, int frameY)
{
	Mat tmp, medianBlurMat;

	medianBlur(eye, medianBlurMat, 7);

	//pokusne orezani oboci
	double eyeTrimHeight = medianBlurMat.size().height * 0.2;
	tmp = medianBlurMat(Rect(0, eyeTrimHeight, medianBlurMat.size().width, medianBlurMat.size().height - (eyeTrimHeight)));
	//tmp = medianBlurMat;

	threshold( tmp, tmp, 18, 255, CV_THRESH_BINARY_INV);
	
	int erosion_size = 1;  // 2
    Mat ErosElement = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size) );
    erode( tmp, tmp, ErosElement );
    //morphologyEx(tmp, tmp, MORPH_OPEN, ErosElement);
    //showWindowAtPosition( windowName + "_open", tmp, x, y + 130);


	/*
    // Create a structuring element
    int dilate_size = 2;  
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * dilate_size + 1, 2 * dilate_size + 1), Point(dilate_size, dilate_size) );
 	// Apply erosion or dilation on the image
 	Mat dilatedMat = tmp.clone();
    //dilate(tmp, dilatedMat, element, Point(-1, -1), 1, BORDER_CONSTANT);
    //showWindowAtPosition( windowName + "_dilate", dilatedMat, x, y + 260);
    */

	vector<vector<Point> > contours;
    Mat threshold_output = tmp.clone();
	vector<Vec4i> hierarchy;

    findContours( threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );

	for( int i = 0; i < contours.size(); i++ )
    { 
    	approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       	boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       	minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }

	//for( int i = 0; i< contours.size(); i++ )
    {
    	//if (boundRect[i].size().height > 4 && boundRect[i].size().width > 2)
    	{
	    	Point frameCenter(center[0].x + frameX, center[0].y + frameY + eyeTrimHeight);
			eyesCentres.push_back(frameCenter);
		}	
    }

    /*
	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
    {
	    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	    drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	    //rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
	    //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );

	    // circle outline
	    Point frameCenter(center[i].x + frameX, center[i].y + frameY + eyeTrimHeight);
		circle( frame, frameCenter, 3, Scalar(0,0,255), 3, 8, 0 );	

	    Point framePoint1(boundRect[i].tl().x + frameX, boundRect[i].tl().y + frameY + eyeTrimHeight);
	    Point framePoint2(boundRect[i].br().x + frameX, boundRect[i].br().y + frameY + eyeTrimHeight);
		rectangle( frame, framePoint1, framePoint2, Scalar(0,255,255), 2, 8, 0 );
    }


    showWindowAtPosition( windowName, drawing, x, y );
    */

	// Vraceni mensi ocni oblasti
	// cout << "size = " << boundRect.size() << endl;
    if (boundRect.size() > 0)
    {
    	cvtColor(eye, eye, CV_GRAY2BGR);
    	int width = eye.size().width * 0.75;
    	//cout << "rozmery zmenseneho oka = " << eye.size().width << ", " << eye.size().height << endl;
    	// cout << "souradnice noveho oka = " << center[0].x - (width * 0.5) << ", " << center[0].y - (width * 0.5) + eyeTrimHeight << endl;
    	// cout << "center oka = " << center[0].x << ", " << center[0].y << endl;

    	Rect newRectEye = Rect(center[0].x - width * 0.5, center[0].y - width * 0.5  + eyeTrimHeight, width, width);
    	if (newRectEye.x < 0)
    		newRectEye.x = 0;
    	else if (newRectEye.x > eye.size().width)
    		newRectEye.x = eye.size().width;

    	if (newRectEye.y < 0)
    		newRectEye.y = 0;
    	else if (newRectEye.y > eye.size().height)
    		newRectEye.y = eye.size().height;
    	//cout << "rozmery zmenseneho oka = " << width << ", " << width << endl;


    	rectangle( eye, newRectEye, Scalar(0,255,255), 2, 8, 0 );

    	
    	//showWindowAtPosition( windowName + "_nova oblast", eye, x, y + 130 );
    	//return Rect(0, 0, 0, 0);
    	return newRectEye;
    }
    else
    {
    	return Rect(0, 0, 0, 0);
    }
}

Mat removeReflections(Mat eye, string windowName, int x, int y, int frameX, int frameY)
{
	Mat gaussEye, binaryEye;

	GaussianBlur( eye, gaussEye, Size(3,3), 0, 0, BORDER_DEFAULT );

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, grad;
	
	int scale = 1;
  	int delta = 0;
  	int ddepth = CV_16S;

	/// Gradient X
	Sobel( gaussEye, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( gaussEye, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );

	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	threshold(grad, binaryEye, HCParam1, 255, CV_THRESH_BINARY);

	//	Mozna jeste pridat erozi???


	Mat reparedEye;
	inpaint(eye, binaryEye, reparedEye, 3, INPAINT_TELEA);

	// Draw
	// showWindowAtPosition( windowName + " - eye", eye, x, y + 260);
	// showWindowAtPosition( windowName + " - bez odlesku", reparedEye, x, y + 390);
	
	return reparedEye;
}

void myCircleHough(Mat eye, string windowName, int x, int y, int frameX, int frameY, Point center)
{
	
	GaussianBlur( eye, eye, Size(3,3), 0, 0, BORDER_DEFAULT );

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, grad;
	
	int scale = 1;
  	int delta = 0;
  	int ddepth = CV_16S;

	/// Gradient X
	Sobel( eye, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( eye, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );

	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	// My houdg with center

	// polomery
	int minRadius = 8, maxRadius = eye.size().width * 0.25	;

	int gradientsCount = maxRadius - minRadius + 1;
	double gradients[gradientsCount];

	for (int i = 0; i < gradientsCount; ++i)
	{
		gradients[i] = 0;
	}

	int i = 0;
	for (int r = minRadius; r <= maxRadius; ++r)
	{
		double step = 2* M_PI / (r*2);

		int stepsCount = 0;
		for(double theta = 0;  theta < 2 * M_PI;  theta += step)
		{
			int x = lround(center.x + r * cos(theta));
			int y = lround(center.y - r * sin(theta));
		
			gradients[i] += grad.at<uchar>(x,y);

			// TMP
			// Vec3b & color = frame.at<Vec3b>(y,x);
			// color[0] = color[1] = 0;
			// color[2] = 255;

			stepsCount++;
		}

		gradients[i] /= stepsCount;
		i++;
	}

	double maxGrad = 0;
	double maxGradRad = 0;
	for (int i = 0; i < gradientsCount; ++i)
	{
		//cout << "Grad = " << gradients[i] << endl;

		if (gradients[i] > maxGrad)
		{
			maxGrad = gradients[i];
			maxGradRad = i;
		}
	}

	maxGradRad += minRadius;

	cout << "max grad = " << maxGrad << " s rad = " << maxGradRad << endl;

	// drawing
	showWindowAtPosition( windowName + "_nova oblast", grad, x, y);

	cvtColor(grad, grad, CV_GRAY2BGR);

	// min max radius circle
	circle( grad, center, minRadius, CV_RGB(0, 0, 255));
	circle( grad, center, maxRadius, CV_RGB(0, 0, 255));

	circle( grad, center, maxGradRad, CV_RGB(255, 0, 0));

	//frame circle
	if (drawInFrame)
	{
		Point frameCenter(center.x + frameX, center.y + frameY);
		circle( frame, frameCenter, maxGradRad, CV_RGB(255, 0, 0), 2);
	}

	showWindowAtPosition( windowName + "_nova oblast + cicles", grad, x, y + 130  );
}


void drawIrises()
{
	bool drawOnlyCorrectIrises = 0;

	if (drawOnlyCorrectIrises)
	{
		for( size_t i = 0; i < irises.size(); i++ )
		{
			int radius = cvRound(irises[i][2]);

			Point frameCenter( cvRound(irises[i][0]), cvRound(irises[i][1]) );
			// circle outline
			Scalar color = Scalar(255, 0, 255);

			circle( frame, frameCenter, radius, color, 3, 8, 0 );	
		}
	}
	else
	{
		for( size_t i = 0; i < eye1.size(); i++ )
		{
			int radius = cvRound(eye1[i][2]);

			Point frameCenter( cvRound(eye1[i][0]), cvRound(eye1[i][1]) );
			// circle outline
			Scalar color = Scalar(0, 0, 255);

			if (eye1[i][5] < 0)
			{
				color = Scalar(255, 0, 255);
				//circle( frame, frameCenter, radius, color, 3, 8, 0 );	
			}

			circle( frame, frameCenter, radius, color, 3, 8, 0 );	
		}

		for( size_t i = 0; i < eye2.size(); i++ )
		{
			int radius = cvRound(eye2[i][2]);

			Point frameCenter( cvRound(eye2[i][0]), cvRound(eye2[i][1]) );
			// circle outline
			Scalar color = Scalar(0, 0, 255);

			if (eye2[i][5] < 0)
			{
				color = Scalar(255, 0, 255);
				//circle( frame, frameCenter, radius, color, 3, 8, 0 );	
			}

			circle( frame, frameCenter, radius, color, 3, 8, 0 );	
		}
	}
}

void drawEyesCentres()
{
	for( size_t i = 0; i < eyesCentres.size(); i++ )
	{
		int radius = 3;

		// circle outline
		Scalar color = Scalar(0, 0, 255);

		//circle( frame, eyesCentres[i], radius, color, 3);
		int lineLength = 10;
		line(frame, Point(eyesCentres[i].x - lineLength*0.5, eyesCentres[i].y), Point(eyesCentres[i].x + lineLength*0.5, eyesCentres[i].y), color);
		line(frame, Point(eyesCentres[i].x, eyesCentres[i].y - lineLength*0.5), Point(eyesCentres[i].x, eyesCentres[i].y + lineLength*0.5), color);
	}
}

void pickCorrectIrises()
{
	int distanceTresh = 5;
	for (int j = 0; j < eyesCentres.size(); ++j)
	{
		for (int i = 0; i < eye1.size(); ++i)
		{	
			double distance = sqrt( pow(eyesCentres[i].x - eye1[i][0], 2) + pow(eyesCentres[i].y - eye1[i][1], 2) );

			if (distance < distanceTresh)
			{
				eye1[i][5] = -1;			 

				irises.push_back(Vec3f(eye1[i][0], eye1[i][1], eye1[i][2]));	
			}			
		}

		for (int i = 0; i < eye2.size(); ++i)
		{	
			double distance = sqrt( pow(eyesCentres[i].x - eye2[i][0], 2) + pow(eyesCentres[i].y - eye2[i][1], 2) );

			if (distance < distanceTresh)
			{
				eye2[i][5] = -1;			 	

				irises.push_back(Vec3f(eye2[i][0], eye2[i][1], eye2[i][2]));
			}
		}
	}


	/*for (int i = 0; i < eye1.size(); ++i)
	{
		for (int j = 0; j < eye2.size(); ++j)
		{
			float tresh = 0.1;	// hodnota se muze lisiti o +/- tresh*100%

			// if ( abs(eye1[i][1] - eye2[j][1]) < 5 && abs(eye1[i][2] == eye2[j][2]) < 10 )
			// {
			// 	eye1[i][5] = -1;
			//  	eye2[j][5] = -1;				
			// }

			double minRadius = 10, maxRadius = 20;
			//cout << "Radius = " << eye1[i][2] << ", " << eye2[j][2] << endl;
			//if ( minRadius < eye1[i][2] < maxRadius &&  minRadius < eye1[j][2] <maxRadius )
			{
				double yMax = 0, yMin = 0, yTresh = 0.9;
				if (eye1[i][1] > eye1[j][1])
				{
					yMax = eye1[i][1];
					yMin = eye1[j][1];
				}
				else
				{
					yMax = eye1[j][1];
					yMin = eye1[i][1];
				}

				double radMax = 0, radMin = 0, radTresh = 0.9;
				if (eye1[i][2] > eye1[j][2])
				{
					radMax = eye1[i][2];
					radMin = eye1[j][2];
				}
				else
				{
					radMax = eye1[j][2];
					radMin = eye1[i][2];
				}

				if ( (yMin / yMax) >= yTresh && (radMin / radMax) >= radTresh )
				{
					eye1[i][5] = -1;
				  	eye2[j][5] = -1;	
				}
			}


			// if ( (eye1[i][3] * (1 - tresh)) > eye2[j][3] > (eye1[i][3] * (1 + tresh)) && (eye1[i][4] * (1 - tresh)) > eye2[j][4] > (eye1[i][4] * (1 + tresh)) )
			// {
			// 	cout << "podobna relativni pozice" << endl;
			// 	eye1[i][5] = -1;
			// 	eye2[j][5] = -1;
			// }

			// if ( abs(eye1[i][1] - eye2[j][1]) < 5 )
			// {
			// 	eye1[i][5] = -1;
			//  	eye2[j][5] = -1;
			// 	cout << "podobna y souradnice" << endl;
			// }

			// if (allEyes[i][2] == allEyes[j][2])
			// {
			// 	//cout << "stejny polomer" << endl;
			// 	// /allEyes[i][5] = -1;
			// }

			// if (allEyes[i][3] == allEyes[j][3])
			// {
			// 	//cout << "stejna relativni x pozice" << endl;
			// 	//allEyes[i][5] = -1;
			// }

			// if (allEyes[i][4] == allEyes[j][4])
			// {
			// 	//cout << "stejna relativni y pozice" << endl;
			// 	//allEyes[i][5] = -1;
			// }				
		}
	}*/
}

void pupilCenterDetection ( Mat eye, string windowName, int x, int y, int frameX, int frameY)
{
	Mat binaryEye;
	double otsu_thresh_val = threshold( eye, binaryEye, 40, 255, CV_THRESH_BINARY);

	int erosion_type, erosion_elem = 2;


	if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
	else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
	else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

	int erosion_size = HCParam2;
  	Mat element = getStructuringElement( erosion_type, Size( 2*erosion_size + 1, 2*erosion_size+1 ), Point( erosion_size, erosion_size ) );

	Mat dilatedEye;
	dilate(binaryEye, dilatedEye, element, Point(-1, -1), 1);	//2

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