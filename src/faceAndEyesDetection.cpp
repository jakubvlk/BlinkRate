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

Point setEyesCentres ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
void myHoughCircle(Mat eye, int kernel, string windowName, int x, int y, int frameX, int frameY, Point center);
Mat removeReflections(Mat eye, string windowName, int x, int y, int frameX, int frameY);

void FCD(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);
void findPupil(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY);

//trackbars
void onHCParam1Trackbar(int pos, void *);
void onHCParam2Trackbar(int pos, void *);
void onHCDpTrackbar(int pos, void *);
void onHCMinDistanceTrackbar(int pos, void *);



// default values
String face_cascade_name = "../../../res/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "../../../res/haarcascade_eye_tree_eyeglasses.xml";
//String face_cascade_name = "../res/haarcascade_frontalface_alt.xml";
//String eyes_cascade_name = "../res/haarcascade_eye_tree_eyeglasses.xml";

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
vector<Vec3f> irises;

int main( int argc, const char** argv )
{
	processArguments( argc, argv);

	CvCapture* capture;
	// Mat frame;

	sliderHCParam1 = HCParam1 = 14;	//26	//35
	sliderHCParam2 = HCParam2 = 14;		//21	//30
	
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
            
            // Pokus - Zkouska, jestli equalize na ocni oblast, zlepsi kvalitu rozpoznani
            Mat newFaceROI = originalFrame(Rect(face.x + eyes[j].x, face.y + eyes[j].y, eyes[j].width, eyes[j].height));
            // convert from color to grayscale
            cvtColor( newFaceROI, newFaceROI, CV_BGR2GRAY );
            // contrast adjustment using the image's histogram
            equalizeHist( newFaceROI, newFaceROI );

            Mat eyeMat = newFaceROI;//faceROI(Rect(eyes[j].x, eyes[j].y, eyes[j].width, eyes[j].height));

			char numstr[21]; // enough to hold all numbers up to 64-bits
			sprintf(numstr, "%d", static_cast<int>(j + 1));
			string eyeName = "eye";

			Mat eyeWithoutReflection = removeReflections(eyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);

			Point eyeCenter = setEyesCentres(eyeWithoutReflection, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);

			myHoughCircle(eyeWithoutReflection, 3, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y, eyeCenter);
			//FCD(eyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
            //findPupil(eyeMat, eyeName + numstr, 820 + 220 * j, 0, face.x + eyes[j].x, face.y + eyes[j].y);
     	}

		if (drawInFrame)
		{
	     	drawEyesCentres();
            drawIrises();
	    }
    }

    //FCD(frame_gray, "", 820 + 220 , 0, 0, 0);
    //findPupil(frame_gray, "", 820 + 220 , 0, 0, 0);


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
	for (int i = 0; i < correctEyes.size(); ++i)
	{
		if (correctEyes[i].y > (face.size().height * 0.5 ))
		{
            cout << "Mazu! Oblast oka mimo vrchni polovinu obliceje. x,y = " << correctEyes[i].x << ", " << correctEyes[i].y << ". Polovina obliceje ma delku " << face.size().height * 0.5 << endl;
			correctEyes.erase(correctEyes.begin() + i);
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
                            cout << "mazu j " << j << endl;
                            
                            if ( i >= correctEyes.size())
                                i = correctEyes.size() - 1;
                            if ( j >= correctEyes.size())
                                j = correctEyes.size() - 1;
						}
						else
						{
							correctEyes.erase(correctEyes.begin() + i);
                            cout << "mazu i " << i << endl;
                            
                            if ( j >= correctEyes.size())
                                j = correctEyes.size() - 1;
                            if ( i >= correctEyes.size())
                                i = correctEyes.size() - 1;
                            
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

Point setEyesCentres ( Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
    //showWindowAtPosition( windowName + " eye centres", eye, windowX, windowY);
    
	Mat tmp, medianBlurMat;

	medianBlur(eye, medianBlurMat, 7);

	//pokusne orezani oboci
	double eyeTrimHeight = medianBlurMat.size().height * 0.2;
	tmp = medianBlurMat(Rect(0, eyeTrimHeight, medianBlurMat.size().width, medianBlurMat.size().height - (eyeTrimHeight)));
	
	int erosion_size = 1;  // 2
    Mat ErosElement = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size) );
    erode( tmp, tmp, ErosElement );
    
    threshold( tmp, tmp, 4, 255, CV_THRESH_BINARY_INV);  // 8
    //showWindowAtPosition( windowName + " tresh", tmp, windowX, windowY + 130);

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
//    showWindowAtPosition( windowName + " eye refl", eye, x, y);
    
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

	threshold(grad, binaryEye, 91, 255, CV_THRESH_BINARY);    //
    //showWindowAtPosition( windowName + " eye bin", binaryEye, x, y);
	//	Mozna jeste pridat erozi???


	Mat reparedEye;
	inpaint(eye, binaryEye, reparedEye, 3, INPAINT_TELEA);

	// Draw
	//showWindowAtPosition( windowName + " - eye", eye, x, y + 260);
	//showWindowAtPosition( windowName + " - bez odlesku", reparedEye, x, y + 390);
	
	return reparedEye;
}

Mat mat2gray(const cv::Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

    return dst;
}

Mat orientationMap(const cv::Mat& mag, const cv::Mat& ori, double thresh = 1.0)
{
    Mat oriMap = Mat::zeros(ori.size(), CV_8UC3);
    Vec3b red(0, 0, 255);
    Vec3b cyan(255, 255, 0);
    Vec3b green(0, 255, 0);
    Vec3b yellow(0, 255, 255);
    for(int i = 0; i < mag.rows*mag.cols; i++)
    {
        float* magPixel = reinterpret_cast<float*>(mag.data + i*sizeof(float));
        if(*magPixel > thresh)
        {
            float* oriPixel = reinterpret_cast<float*>(ori.data + i*sizeof(float));
            Vec3b* mapPixel = reinterpret_cast<Vec3b*>(oriMap.data + i*3*sizeof(char));
            if(*oriPixel < 90.0)
                *mapPixel = red;
            else if(*oriPixel >= 90.0 && *oriPixel < 180.0)
                *mapPixel = cyan;
            else if(*oriPixel >= 180.0 && *oriPixel < 270.0)
                *mapPixel = green;
            else if(*oriPixel >= 270.0 && *oriPixel < 360.0)
                *mapPixel = yellow;
        }
    }

    return oriMap;
}

void FCD(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
    int64 e1 = getTickCount();
    
	GaussianBlur( eye, eye, Size(3,3), 0, 0, BORDER_DEFAULT );

	// Gradient
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, grad;
	
	int scale = 1;
  	int delta = 0;
  	int ddepth = CV_32F;

	/// Gradient X
	Sobel( eye, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( eye, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );

	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );


    Mat direction, acceptedDirectionPares;
    
    acceptedDirectionPares = eye.clone();
    // Obarveni na cerno - to by asi slo udelat lip - napr. vytvorit Mat se stejnymi rozmery jako grad, ale rovnou cerny, nebo tak neco...
    for (int i = 0; i < acceptedDirectionPares.cols; i++)
    {
        for (int j = 0; j < acceptedDirectionPares.rows; j++)
        {
            acceptedDirectionPares.at<uchar>(i, j) = 0;
        }
    }

    phase(grad_x, grad_y, direction, true);
    
    double m_pi180 = M_PI / 180;
    double _180m_pi = 180 / M_PI;
    
    vector<Vec4f> pairVectors;
    
    for (int i = 0; i < direction.cols; i++)
    {
        for (int j = 0; j < direction.rows; j++)
        {
            for (int k = i; k < direction.cols; k++)
            {
                for (int l = j + 1; l <= direction.rows; l++)
                {
//            for (int k = 0; k < direction.cols; k++)
//            {
//                for (int l = 0; l < direction.rows; l++)
//                {
                    // Pokud nemaji nulovou velikost gradientu
                    if (abs_grad_x.at<uchar>(i, j) != 0 && abs_grad_x.at<uchar>(k, l) != 0)
                    {
                        int opositeAngleTresh = 4;
                        // musi mit cca opacny uhel
                        if( abs( direction.at<float>(i, j) - direction.at<float>(k, l) ) > ( 180 - opositeAngleTresh) &&
                           abs( direction.at<float>(i, j) - direction.at<float>(k, l) ) < ( 180 + opositeAngleTresh) )
                        {
                            float rad1 = direction.at<float>(i, j) * m_pi180;
                            Vec2f directionVec1 = Vec2f( cos(rad1), sin(rad1));
                            
//                            float rad2 = direction.at<float>(k, l) * m_pi180;
//                            Vec2f directionVec2 = Vec2f( cos(rad2), sin(rad2));
//                            
//                            Vec2f p1p2 = directionVec1 - directionVec2;
//                            
//                            float angle = acos( ( directionVec1[0] * p1p2[0] + directionVec1[1] * p1p2[0] ) / (sqrt(pow(directionVec1[0], 2) + pow(directionVec1[1], 2))) * (sqrt(pow(p1p2[0], 2) + pow(p1p2[1], 2))) );
                            
                            
                            // ***************************************
                            // pixely - alternative 2
//                            Point p1p2_B = Point(l - j,k - i);
                            Point p1p2_B = Point(j - l,i - k);
                            
                            float magnitude = sqrt(pow(p1p2_B.x, 2) + pow(p1p2_B.y, 2));
                            Point2f p1p2Normalized_B = Point2f( p1p2_B.x / magnitude, p1p2_B.y / magnitude );
                            
                            Point2f v1 = Point2f(directionVec1[0], directionVec1[1]);
                            
                            float angle_B = acos( v1.dot(p1p2Normalized_B) / (sqrt ( pow(v1.x, 2) + pow(v1.y, 2)) * sqrt(pow(p1p2Normalized_B.x, 2) + pow(p1p2Normalized_B.y, 2))) ) * _180m_pi;
                            
                            //cout << "angle = " << angle_B << endl;
                            
                            // Uhel mezi p1p2 a v1 kolem 0
                            int angleTresh = 4;  // 4
                            if (abs(angle_B) < angleTresh)
                            {
                                pairVectors.push_back(Vec4f(j, i, l, k));
                                
                                // debug obarveni
                                acceptedDirectionPares.at<uchar>(i, j) = 255;
                                acceptedDirectionPares.at<uchar>(k, l) = 255;
//                                acceptedDirectionPares.at<uchar>(l, k) = grad.at<uchar>(l, k);
//                                acceptedDirectionPares.at<uchar>(j, i) = grad.at<uchar>(j, i);
                            }
                        }
                        
                        //acceptedDirectionPares.at<uchar>(k, l) = grad.at<uchar>(k, l);
                    }
                }
            }
        }
    }
    
    // *********************** TMP **************************
    //25,11 ; 46,36
    //pairVectors.clear();
    //pairVectors.push_back(Vec4f(25, 11, 46, 36));
    
    vector<Vec4f> pairVectorsWithRadius;
    int minRad = 5, maxRad = 20;
    
    for (int i = 0; i < pairVectors.size(); i++)
    {
        Vec2f vec = Vec2f(pairVectors[i][0] - pairVectors[i][2], pairVectors[i][1] - pairVectors[i][3]);
        
        float mag = sqrt(pow(vec[0], 2) + pow(vec[1], 2));
        
        if (mag > minRad*2 && mag < maxRad*2)
        {
            pairVectorsWithRadius.push_back(Vec4f(pairVectors[i][0], pairVectors[i][1], pairVectors[i][2], pairVectors[i][3]));
        }
    }
    
    // TODO: zmensit pole jenom na pouzite radiusy. Ted je tam zbytecne 0 az minRadius-1
    int accumulator[abs_grad_x.cols][abs_grad_x.rows][maxRad];
    for (int i = 0; i < abs_grad_x.cols; ++i)
    {
        for (int j = 0; j < abs_grad_x.rows; ++j)
        {
            for (int k = 0; k < maxRad; ++k)
            {
                accumulator[i][j][k] = 0;
            }
        }
    }
    
    for (int i = 0; i < pairVectorsWithRadius.size(); ++i)
    {
        // *********************** TMP **************************
        //25,11 ; 46,36
        //acceptedDirectionPares.at<uchar>(pairVectorsWithRadius[i][1], pairVectorsWithRadius[i][0]) = 255;
        //acceptedDirectionPares.at<uchar>(pairVectorsWithRadius[i][3], pairVectorsWithRadius[i][2]) = 255;
        
        Vec2f vec = Vec2f(abs(pairVectorsWithRadius[i][0] - pairVectorsWithRadius[i][2]), abs(pairVectorsWithRadius[i][1] - pairVectorsWithRadius[i][3]));   // abs??
        //acceptedDirectionPares.at<uchar>(vec[1], vec[0]) = 255;
        
        float mag = sqrt(pow(vec[0], 2) + pow(vec[1], 2));
//        Vec2i center = Vec2i(lround(vec[0] * 0.5), lround(vec[1] * 0.5));
//        center += Vec2i(pairVectorsWithRadius[i][0], pairVectorsWithRadius[i][1]);
        Vec2i center = Vec2f((pairVectorsWithRadius[i][0] + pairVectorsWithRadius[i][2]), (pairVectorsWithRadius[i][1] + pairVectorsWithRadius[i][3])) * 0.5;
        
        if (center[0] >= abs_grad_x.cols)
            center[0] = abs_grad_x.cols -1;
        if (center[1] >= abs_grad_x.rows)
            center[1] = abs_grad_x.rows -1;
        
        accumulator[center[0]][center[1]][lround(mag * 0.5)]++;
        //acceptedDirectionPares.at<uchar>(center[1], center[0]) = 255;
    }
    
//    for (int i = 0; i < grad.cols; ++i)
//    {
//        for (int j = 0; j < grad.rows; ++j)
//        {
//            for (int k = minRad; k < maxRad; ++k)
//            {
//                if (accumulator[i][j][k] != 0)
//                    cout << i << ", " << j << ", " << k << " = " << accumulator[i][j][k] << endl;
//            }
//        }
//    }
    
    Vec3i bestCircle;
    int max = 0;
    for (int i = 0; i < abs_grad_x.cols; ++i)
    {
        for (int j = 0; j < abs_grad_x.rows; ++j)
        {
            for (int k = minRad; k < maxRad; ++k)
            {
                if (accumulator[i][j][k] > max)
                {
                    max = accumulator[i][j][k];
                    bestCircle = Vec3i(i, j, k);
                }
            }
            
            //circle(frame, Point(bestCircle[0], bestCircle[1]), bestCircle[2], CV_RGB(0, 0, 255));
        }
    }
    cout << "best cirlce is " << bestCircle << " with max = " << max << endl;
    circle(frame, Point(bestCircle[0] + frameX, bestCircle[1] + frameY), bestCircle[2], CV_RGB(255, 0, 0));
    
//    // zobrazeni akumulatoru
//    Mat accMat = grad.clone();
//    for (int i = 0; i < accMat.cols; ++i)
//    {
//        max = 0;
//        for (int j = 0; j < accMat.rows; ++j)
//        {
//            for (int k = minRad; k < maxRad; ++k)
//            {
//                if (accumulator[i][j][k] > max)
//                {
//                    max = accumulator[i][j][k];
//                    bestCircle = Vec3i(i, j, k);
//                }
//            }
//            
//            accMat.at<uchar>(j, i) = max;
//        }
//    }
    
    
    double time = (getTickCount() - e1)/ getTickFrequency();
    cout << endl << "time = " << time << endl;

	showWindowAtPosition( windowName + "grad", abs_grad_x, windowX, windowY);
    showWindowAtPosition( windowName + "_direction", mat2gray(direction), windowX, windowY + 130);
    showWindowAtPosition( windowName + "_accepted dir", acceptedDirectionPares, windowX, windowY + 260);
    //showWindowAtPosition( windowName + "_acc", accMat, windowX, windowY + 390);
    
}

void findPupil(Mat eye, string windowName, int windowX, int windowY, int frameX, int frameY)
{
    Mat tresholdedEye;
    
    //eye = frame = imread("/Users/jakubvlk/MyShit/BlinkRate/res/pics/gradTestSmallEye.jpg");
    cvtColor( eye, eye, CV_BGR2GRAY );
    
    // contrast adjustment using the image's histogram
    equalizeHist( eye, eye );
    
    threshold( eye, tresholdedEye, HCParam1, 255, CV_THRESH_BINARY);
    
    //showWindowAtPosition( windowName + " eye", eye, windowX, windowY + 520);
    //showWindowAtPosition( windowName + " tresh", tresholdedEye, windowX, windowY + 600);
}

void myHoughCircle(Mat eye, int kernel, string windowName, int windowX, int windowY, int frameX, int frameY, Point center)
{
    //showWindowAtPosition( windowName + "PRE eye hough", eye, windowX, windowY );
    
	GaussianBlur( eye, eye, Size(7,7), 0, 0, BORDER_DEFAULT );

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
    
    // Je to lepsi???????????????
    grad = mat2gray(abs_grad_x);


	// polomery
	int minRadius = 8, maxRadius = eye.size().width * 0.3	;

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
        xMin = xMax = center.x;
        yMin = yMax = center.y;
        
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
                    
					gradients[i][j][k] += grad.at<uchar>(circleY, circleX);
                    
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
		for (int y = yMin; y <= yMax; ++y)
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

    //cout << "max grad = " << maxGrad << " s rad = " << maxGradRad << endl << endl;
	
	// drawing
	showWindowAtPosition( windowName + "_nova oblast", grad, windowX, windowY + 130);

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


	showWindowAtPosition( windowName + "_nova oblast + cicles", grad, windowX, windowY + 260 );


	cvtColor(eye, eye, CV_GRAY2BGR);
	circle(eye, center, maxGradRad, color);
	//showWindowAtPosition( windowName + "_nova oblast + eye", eye, windowX, windowY + 260  );
    
    showWindowAtPosition( windowName + "POST eye hough", eye, windowX, windowY + 390);
    //findPupil(eye(Rect(center.x - maxGradRad, center.y - maxGradRad, maxGradRad*2, maxGradRad*2)), windowName, windowX, windowY, frameX, frameY);
    //findPupil(frame(Rect(frameX + center.x - maxGradRad, frameY + center.y - maxGradRad, maxGradRad*2, maxGradRad*2)), windowName, windowX, windowY, frameX, frameY);
    //showWindowAtPosition( windowName +  + "eye", frame(Rect(frameX, frameY, eye.size().width, eye.size().height)), windowX, windowY + 650);
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