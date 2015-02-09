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

void showWindowAtPosition( string imageName, Mat mat, int x, int y );
void refreshImage();
vector<Rect> pickEyeRegions(vector<Rect> eyes, Mat face);
Rect pickFace(vector<Rect> faces);


void drawIrises();
void drawEyesCentres();

Point setEyesCentres ( Mat eye, string windowName, int x, int y, int frameX, int frameY);
void myCircleHough(Mat eye, int kernel, string windowName, int x, int y, int frameX, int frameY, Point center);
Mat removeReflections(Mat eye, string windowName, int x, int y, int frameX, int frameY);


//trackbars
void onHCParam1Trackbar(int pos, void *);
void onHCParam2Trackbar(int pos, void *);
void onHCDpTrackbar(int pos, void *);
void onHCMinDistanceTrackbar(int pos, void *);



// default values
String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "../res/haarcascade_eye_tree_eyeglasses.xml";
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

	sliderHCParam1 = HCParam1 = 18;	//26	//35
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

			Point eyeCenter = setEyesCentres(eyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);

			myCircleHough(eyeMat, 3, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeCenter);
     	}

		if (drawInFrame)
		{
	     	drawIrises();
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
        //int faceSize =
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
				double distance = /*sqrt*/( pow(correctEyes[i].x - correctEyes[j].x, 2.) + pow(correctEyes[i].y - correctEyes[j].y, 2.) );

				if (face.size().width != 0)
				{
					//cout << "distance = " << distance / face.size().width << endl;
					if (distance / pow( face.size().width, 2.) < distancesTresh)
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

Point setEyesCentres ( Mat eye, string windowName, int x, int y, int frameX, int frameY)
{
	Mat tmp, medianBlurMat;

	medianBlur(eye, medianBlurMat, 7);

	//pokusne orezani oboci
	double eyeTrimHeight = medianBlurMat.size().height * 0.2;
	tmp = medianBlurMat(Rect(0, eyeTrimHeight, medianBlurMat.size().width, medianBlurMat.size().height - (eyeTrimHeight)));
	
	threshold( tmp, tmp, 8, 255, CV_THRESH_BINARY_INV);
	
	int erosion_size = 1;  // 2
    Mat ErosElement = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size) );
    erode( tmp, tmp, ErosElement );
   

	vector<vector<Point> > contours;
    Mat threshold_output = tmp.clone();
	vector<Vec4i> hierarchy;

    findContours( threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly( contours.size() );
	//vector<Rect> boundRect( contours.size() );
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );

	for( int i = 0; i < contours.size(); i++ )
    { 
    	approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       	//boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       	minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    }

    Point correctCenter = Point(tmp.size().width * 0.5, tmp.size().height * 0.5);
    // Pokud najdeme vice kontur, tak nechame jen tu nejvetsi
    if (contours.size() > 0)
    {
	    correctCenter = center[0];
	    if (contours.size() > 1)
	    {
	    	int maxRadius = 0;
	    	int maxRadiusIndex = 0;

	    	for (int i = 0; i < contours.size(); ++i)
	    	{
	    		if (radius[i] > maxRadius)
		    	{
		    		maxRadius = radius[i];
		    		maxRadiusIndex = i;
		    	}	
	    	}

	    	correctCenter = center[maxRadiusIndex];
	    }



		Point frameCenter(correctCenter.x + frameX, correctCenter.y + frameY + eyeTrimHeight);
		eyesCentres.push_back(frameCenter);
	}

    return Point(correctCenter.x, correctCenter.y + eyeTrimHeight);
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

	threshold(grad, binaryEye, 180, 255, CV_THRESH_BINARY);

	//	Mozna jeste pridat erozi???


	Mat reparedEye;
	inpaint(eye, binaryEye, reparedEye, 3, INPAINT_TELEA);

	// Draw
	// showWindowAtPosition( windowName + " - eye", eye, x, y + 260);
	// showWindowAtPosition( windowName + " - bez odlesku", reparedEye, x, y + 390);
	
	return reparedEye;
}

void myCircleHough(Mat eye, int kernel, string windowName, int windowX, int windowY, int frameX, int frameY, Point center)
{

	GaussianBlur( eye, eye, Size(3,3), 0, 0, BORDER_DEFAULT );

	// Gradient
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


	// polomery
	int minRadius = 8, maxRadius = eye.size().width * 0.25	;

	int gradientsCount = maxRadius - minRadius + 1;
	double gradients[kernel][kernel][gradientsCount];

	// nulovani
	for (int i = 0; i < kernel; ++i)
	{
		for (int j = 0; j < kernel; ++j)
		{
			for (int k = 0; k < gradientsCount; ++k)
			{
				gradients[i][j][k] = 0;
			}
		}
	}

	int newKernel = (kernel - 1) * 0.5;
	// hranice
	int xMin = center.x - newKernel;
	if (xMin < 0)
		xMin = 0;
	int xMax = center.x + newKernel;
	if (xMax > eye.size().width)
		xMax = eye.size().width;
	int yMin = center.y - newKernel;
	if (yMin < 0)
		yMin = 0;
	int yMax = center.y + newKernel;
	if (yMax > eye.size().height)
		yMax = eye.size().height;

	// cout << "center = " << center.x << ", " << center.y << endl;
	// cout << xMin << ", " << xMax << ", " << yMin << ", " << yMax << endl;
    
    if (kernel == 1)
    {
        xMin = yMin = 0;
        xMax = yMax = 0;
        
    }

	int i = 0;
	int totalStepsCount = 0;
	for (int x = xMin; x <= xMax; ++x)
	{		
		int j = 0;
		for (int y = yMin; y <= yMax; ++y)
		{
			int k = 0;
			for (int r = minRadius; r <= maxRadius; ++r)
			{
				double step = 2* M_PI / (r*2);

				int stepsCount = 0;
				for(double theta = 0;  theta < 2 * M_PI;  theta += step)
				{
					int circleX = lround(x + r * cos(theta));
					int circleY = lround(y - r * sin(theta));
				
					gradients[i][j][k] += grad.at<uchar>(circleX, circleY);

					stepsCount++;
				}

				totalStepsCount += stepsCount;

				gradients[i][j][k] /= stepsCount;
				k++;
			}

			j++;
		}

		i++;
	}
	

	double maxGrad = 0;
	double maxGradRad = 0;
	Point newCenter = center;

	i = 0;
	for (int x = xMin; x <= xMax; ++x)
	{		
		int j = 0;
		for (int y = yMin; y < yMax; ++y)
		{
			for (int k = 0; k < gradientsCount; ++k)
			{		
				if (gradients[i][j][k] > maxGrad)
				{
					maxGrad = gradients[i][j][k];
					maxGradRad = k;

					newCenter.x = x;
					newCenter.y = y;
				}
			}
		}
	}

	

	maxGradRad += minRadius;

	//cout << "max grad = " << maxGrad << " s rad = " << maxGradRad << endl;
	
	// drawing
	//showWindowAtPosition( windowName + "_nova oblast", grad, windowX, windowY);

	cvtColor(grad, grad, CV_GRAY2BGR);

	Scalar color = Scalar(0, 0, 255);
	int lineLength = 10;
	line(grad, Point(center.x - lineLength*0.5, center.y), Point(center.x + lineLength*0.5, center.y), color);
	line(grad, Point(center.x, center.y - lineLength*0.5), Point(center.x, center.y + lineLength*0.5), color);

	// min max radius circle
	circle( grad, center, minRadius, CV_RGB(0, 0, 255));
	circle( grad, center, maxRadius, CV_RGB(0, 0, 255));

	circle( grad, center, maxGradRad, CV_RGB(255, 0, 0));

	irises.push_back(Vec3f(center.x + frameX, center.y + frameY, maxGradRad));


	//showWindowAtPosition( windowName + "_nova oblast + cicles", grad, windowX, windowY + 130 );


	cvtColor(eye, eye, CV_GRAY2BGR);
	circle(eye, center, maxGradRad, color);
	//showWindowAtPosition( windowName + "_nova oblast + eye", eye, windowX, windowY + 260  );
}


void drawIrises()
{
	for( size_t i = 0; i < irises.size(); i++ )
	{
		int radius = cvRound(irises[i][2]);

		Point frameCenter( cvRound(irises[i][0]), cvRound(irises[i][1]) );
		// circle outline
		Scalar color = Scalar(255, 0, 255);

		circle( frame, frameCenter, radius, color, 2);
	}
}

void drawEyesCentres()
{
	for( size_t i = 0; i < eyesCentres.size(); i++ )
	{
		// circle outline
		Scalar color = Scalar(0, 0, 255);

		//circle( frame, eyesCentres[i], radius, color, 3);
		int lineLength = 10;
		line(frame, Point(eyesCentres[i].x - lineLength*0.5, eyesCentres[i].y), Point(eyesCentres[i].x + lineLength*0.5, eyesCentres[i].y), color);
		line(frame, Point(eyesCentres[i].x, eyesCentres[i].y - lineLength*0.5), Point(eyesCentres[i].x, eyesCentres[i].y + lineLength*0.5), color);
	}
}

void showWindowAtPosition( string imageName, Mat mat, int x, int y )
{
	imshow( imageName, mat );
	moveWindow(imageName, x, y);
}