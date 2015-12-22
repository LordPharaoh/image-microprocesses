#include "buoyNew.hpp"

//obtains image; currently uses imread but should be switched to Surya's thing.
cv::Mat obtainImage(int argc, char** argv){
	if(argc != 2){
		std::cout << "Incorrect usage of arguments";
	}
	cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	return(image);
}
 
 
 
//this came from visiontask.cpp in eva 
cv::Mat equalColorHist(cv::Mat& img, bool red, bool green, bool blue)
{
	std::vector<cv::Mat> channels;
	cv::split(img, channels);

	if (blue)
	{
		cv::equalizeHist(channels[0], channels[0]);
	}
	if (green)
	{
		cv::equalizeHist(channels[1], channels[1]);
	}
	if (red)
	{
		cv::equalizeHist(channels[2], channels[2]);
	}

	cv::Mat result;
	cv::merge(channels, result);
	return result;
}
cv::Mat generateDiffMap(cv::Mat& img, int diff)
{
	cv::Mat diffMap = cv::Mat(img.size(), CV_32F, cv::Scalar(0));
	float* ip = img.ptr<float>();
	float* op = diffMap.ptr<float>();
	auto getDiffs = [=](int xBeg, int xEnd, int yBeg, int yEnd)
	{
		for (int c = xBeg; c < xEnd; c++)
		{
			for (int r = yBeg; r < yEnd; r++)
			{
				// make the value equal to how much it stands out from its horizontal or vertical neighbors, whichever is less
				float vert = (2*ip[(r*img.cols+c)]-ip[((r-diff)*img.cols+c)]-ip[((r+diff)*img.cols+c)]);
				float hori = (2*ip[(r*img.cols+c)]-ip[(r*img.cols+c+diff)]-ip[(r*img.cols+c-diff)]);
				float weak = (std::abs(vert) < std::abs(hori)) ? vert : hori;
				op[r*img.cols+c] = weak;
			}
		}
	};
	return diffMap;
}
	
//These values were originally in config for l_gate but i don't see a need for them to be configurable. 
//If you would like them to be configurable, talk to Varun. If Varun has graduated/quit for the Blue Devils by this point, I wish you the best of luck
float cropx = 1.0;
float cropy = 1.0;
float offset = .9 * (1-cropy);
float scalex = 0.2;
float scaley = 0.2;
int diffDist = 8;
//End supposedly config values that I suspect were never acually modified
//
//
int rx,ry,gx,gy,yx,yy;
int main(int argc, char** argv){
	//obtain image; see above
	cv::Mat firstImage = obtainImage(argc, argv);
	//define processedImage
	cv::Mat processedImage;
	
	int image_size = firstImage.rows*firstImage.cols;
	//this paranthesized staement is large enough to justify newlines and indentation
	//It resizes the image by some arcane parameters
	if (firstImage.cols == 0 ){
		std::cout<<"Error reading file";
		return -1;
	}
	firstImage(
		cv::Rect(
			firstImage.cols*(1-cropx)/2,
			firstImage.rows*(1-cropy-offset)/2,
			firstImage.cols*cropx,
			firstImage.rows*cropy
			)
		  );
	//std::cout << "\n" << firstImage.cols << "\n" << firstImage.rows << "\n" << cropx << "\n" << cropy << "\n" << offset;
	cv::Mat yNot;
	cv::resize(
		firstImage,
		yNot,
		cv::Size(
			firstImage.cols*cropx*scalex,
			firstImage.rows*cropy*scaley
			)
		);
	//and finally the end of the monstrosity
	//create a temp image to store lower smaller processed image
	cv::Mat output(yNot.rows, yNot.cols, CV_8UC3, cv::Scalar(0,0,0));
	//makign a pointer to first element of output
	unsigned char* op = output.ptr();
	
	//initialising come Mats for later usage
	cv::Mat imgR, imgG, imgY;
	imgY.create(yNot.rows, yNot.cols, CV_8UC1);
	imgG.create(yNot.rows, yNot.cols, CV_8UC1);
	imgR.create(yNot.rows, yNot.cols, CV_8UC1);
					 
	unsigned char *rPtr = imgR.ptr();
	unsigned char *gPtr = imgG.ptr();
	unsigned char *yPtr = imgY.ptr();
	const unsigned char* ip = yNot.ptr();

	//filter for green
	cv::Mat grMat(yNot.size(), CV_32F, cv::Scalar(0));
	cv::Mat reMat(yNot.size(), CV_32F, cv::Scalar(0));
	float* gp = grMat.ptr<float>();
	float* rp = reMat.ptr<float>();
	cv::Mat histo = equalColorHist(yNot, false, false, false);
	unsigned char* hp = histo.ptr();
	auto filterColors = [=](int beg, int end)
	{
		for (int i = 0; i < yNot.rows*yNot.cols; i++)
		{
			int b = hp[3*i], g = hp[3*i+1], r = (int)hp[3*i+2];
			//gp[i] = 40.0*(g-0.4*r)/(b+3.0*r+10.1);
			gp[i] = (g+0.4*r)/(b+0.7*r+3.01)-0.004*r;
			rp[i] = 70.0*r/(g+1.0) - 120.0*g/(b+1.0)-(g+b)*0.1;
		}
	};
	cv::Mat diMap = generateDiffMap(grMat, diffDist);
	cv::Mat drMap = generateDiffMap(reMat, diffDist);
	float* drffp = drMap.ptr<float>();

	cv::blur(diMap, diMap, cv::Size(3,3));
	float* diffp = diMap.ptr<float>();
	float maxR[4][4] = {{-1000, 0, 0, 1000}};
	int maxY[4][3] = {{-1}};
	float maxG[4][4] = {{-1000, 0, 0, 1000}};
	auto filterRY = [=](int start, int end, float rVals[4], int yVals[3], float gVals[4])
	{
		for (int i = 0; i < yNot.cols*yNot.rows; i++)
		{
			int b = ip[3*i], g = ip[3*i+1], r = (int)ip[3*i+2];
			
			op[3*i+2] = drffp[i]/10+128;
			// filter for red
			if (drffp[i] > rVals[0])
			{
				rVals[0] = drffp[i];
				rVals[1] = i % yNot.cols;
				rVals[2] = i / yNot.cols;
			}
			if (drffp[i] < rVals[3])
			{
				rVals[3] = drffp[i];
			}

			// filter for yellow
			yPtr[i] = std::max(0, std::min(255, 128+(r+g-b)/4));
			op[3*i] = yPtr[i];

			// find yellow buoy
			if (yPtr[i] > yVals[0])
			{
				yVals[0] = yPtr[i];
				yVals[1] = i % yNot.cols;
				yVals[2] = i / yNot.cols;
			}

			op[3*i+1] = diffp[i]+128;
			//find green buoy and max/min values
			if (diffp[i] > gVals[0])
			{
				gVals[0] = diffp[i];
				gVals[1] = i % yNot.cols;
				gVals[2] = i / yNot.cols;
			}
			if (diffp[i] < gVals[3])
			{
				gVals[3] = diffp[i];
			}
		}
	};
	float rMax = -2;
	float rMin = -2;
	int yMax = -2;
	float gMax = -2;
	float gMin = -2;
	for (int i = 0; i < 4; i++)
	{
		if (maxR[i][0] > rMax)
		{
			rMax = maxR[i][0];
			rx = (int)maxR[i][1];
			ry = (int)maxR[i][2];
		}
		if (maxR[i][3] < rMin)
		{
			rMin = maxR[i][3];
		}
		if (maxY[i][0] > yMax)
		{
			yMax = maxY[i][0];
			yx = maxY[i][1];
			yy = maxY[i][2];
		}
		if (maxG[i][0] > gMax)
		{
			gMax = maxG[i][0];
			gx = (int)maxG[i][1];
			gy = (int)maxG[i][2];
		}
		if (maxG[i][3] < gMin)
		{
			gMin = maxG[i][3];
		}
	}

	// highlight buoys on processed yNot
	op[(int)(3*(ry*yNot.cols+rx))] = 0;
	op[(int)(3*(ry*yNot.cols+rx))+1] = 0;
	op[(int)(3*(gy*yNot.cols+gx))] = 0;
	op[(int)(3*(gy*yNot.cols+gx))+2] = 0;
	op[(int)(3*(yy*yNot.cols+yx))+1] = 0;
	op[(int)(3*(yy*yNot.cols+yx))+2] = 0;
	op[(int)(3*(gy*yNot.cols+gx))+1] = 255;
	op[(int)(3*(ry*yNot.cols+rx))+2] = 255;
	op[(int)(3*(yy*yNot.cols+yx))] = 255;

	// resize filtered yNot for processed yNot

	//this stuff didn't work so I removed it
	 /*processedImage(
			cv::Rect(processedImage.cols*(1-cropx)/2, 
			processedImage.rows*(1-cropy-offset)/2, 
			processedImage.cols*cropx,
			processedImage.rows*cropy)
			);*/
	//std::cout << "\n processedImage.rows: " << processedImage.rows << "\n processedImage.cols: " << processedImage.cols; 
	/*cv::resize(output, processedImage, 
		cv::Size(processedImage.cols*cropx, processedImage.rows*cropy), 0, 0, cv::INTER_NEAREST
		);*/
	rx = (rx - yNot.cols/2) / yNot.cols;
	ry = (yNot.rows/2 - ry) / yNot.rows;
	gx = (gx - yNot.cols/2) / yNot.cols;
	gy = (yNot.rows/2 - gy) / yNot.rows+offset;
	yx = (yx - yNot.cols/2) / yNot.cols;
	yy = (yNot.rows/2 - yy) / yNot.rows;
 
	std::cout <<"\n"<<  rx<<"\n"<<  ry<<"\n"<<  gx<<"\n"<<  gy<<"\n"<<  yx<<"\n"<<  yy << "\n";
 
	return(0);
}
 
 
 
 
 
 
 
 
 
 
 
 
