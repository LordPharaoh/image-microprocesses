#include"dropperNew.hpp"

cv::Mat img_enhanced;
static std::vector< std::array<double,2> > black_vals (3);
static std::vector< std::array<double,2> > orange_vals (3);
static std::vector< std::array<double,2> > white_vals (3);
static std::vector< std::array<double,2> > yellow_vals (3);
int yellow_min_area, orange_min_area, black_min_area, white_min_area;
int white_hull_min_area;
int black_fallback;
std::vector<cv::Point2f> bin_contours(cv::Mat img_bgr_enhanced, std::vector<cv::Mat> *masks, std::vector< std::vector<cv::Point> > *hulls)
{
	cv::Mat img_hsv;
	cv::cvtColor(img_bgr_enhanced, img_hsv, CV_BGR2HSV, 0);

	// threshold images by HSV values
	cv::Mat img_threshed_yellow = color_crop::multichannel_thresh(img_hsv, yellow_vals);
	cv::Mat img_threshed_black = color_crop::multichannel_thresh(img_hsv, black_vals);
	cv::Mat img_threshed_white = color_crop::multichannel_thresh(img_hsv, white_vals);
	cv::Mat img_threshed_orange = color_crop::multichannel_thresh(img_hsv, orange_vals);

	// get contours from each threshold
	std::pair<std::vector< std::vector<cv::Point > >, std::vector<cv::Vec4i> >
		contours_white = color_crop::image_to_contours(img_threshed_white, 100);

	std::pair<std::vector< std::vector<cv::Point> >, std::vector<cv::Vec4i> >
		contours_yellow = color_crop::image_to_contours(img_threshed_yellow, 100, 2
				/* higher dilation because the contours here should be pretty
				 * far apart and it's harder to get consistent contours with
				 * this */);

	std::pair<std::vector< std::vector<cv::Point> >, std::vector<cv::Vec4i> >
		contours_black = color_crop::image_to_contours(img_threshed_black, 100);

	std::pair<std::vector< std::vector<cv::Point> >, std::vector<cv::Vec4i> >
		contours_orange = color_crop::image_to_contours(img_threshed_orange, 100);

	// filter contours by area
	std::vector<bool> scores_area_white = color_crop::filter_contours_area(
			contours_white.first, white_min_area);
	std::vector<bool> scores_area_yellow = color_crop::filter_contours_area(
			contours_yellow.first, yellow_min_area);
	std::vector<bool> scores_area_black = color_crop::filter_contours_area(
			contours_black.first, black_min_area);
	std::vector<bool> scores_area_orange = color_crop::filter_contours_area(
			contours_orange.first, orange_min_area);

	std::vector <std::vector<cv::Point> > contours_white_children_valid =
		contours_white.first;

	color_crop::filter(scores_area_white, contours_white_children_valid);

	/* get white contour convex hulls in an attempt to make
	 * color_crop::filter_contours_withit() more reliable */
	std::vector< std::vector<cv::Point> > hulls_white;
	color_crop::contours_to_convexhulls(contours_white_children_valid,
			hulls_white);

	// filter these hulls by area now
	color_crop::filter(color_crop::filter_contours_area(hulls_white, white_hull_min_area), hulls_white);

	// remove white contours that do not contain a black contour
	color_crop::filter(scores_area_black, contours_black.first);

	std::vector<int> indices_white_around_black, indices_black_in_white;
	color_crop::filter_contours_within(hulls_white,
			indices_white_around_black, contours_black.first,
			indices_black_in_white, 0.8);
	hulls_white = color_crop::indices_to_vector(hulls_white,
			indices_white_around_black);

	// remove white contours that don't have a yellow within
	color_crop::filter(scores_area_yellow, contours_yellow.first);

	std::vector<int> indices_white_proto, indices_yellow_proto;
	color_crop::filter_contours_within(hulls_white, indices_white_proto,
			contours_yellow.first, indices_yellow_proto, 0.8);
	hulls_white = color_crop::indices_to_vector(
			hulls_white, indices_white_proto);

	// remove white contours that contain an orange contour
	// to avoid seeing bin cover as bin
	color_crop::filter(scores_area_orange, contours_orange.first);

	std::vector<int> indices_white_around_orange, indices_orange_in_white;
	color_crop::filter_contours_within(hulls_white,
			indices_white_around_orange, contours_orange.first,
			indices_orange_in_white, 0.8);
	color_crop::filter(color_crop::indices_to_scores(
				indices_white_around_orange, hulls_white.size()),
			hulls_white, true);

	// remove white contours that contain another white contour
	// this is done because when the bins are clone or overlapping,
	// there might be one big contour around both bins. we want to remove
	// that contour
	color_crop::filter(color_crop::filter_out_enclosing_contours(
				hulls_white, .51), hulls_white);


	if ((hulls_white.size() < sizeof(bins)/sizeof(bins[0])) && black_fallback)
	{
		// as a last ditch effort, we will look for valid black hulls that
		// are not contained by a white hull and we will add those in too

		color_crop::filter(scores_area_black, contours_black.first);
		std::vector< std::vector<cv::Point> > hulls_black;
		color_crop::contours_to_convexhulls(contours_black.first, hulls_black);

		// look for black hulls with yellow contours within
		std::vector<int> indices_black_proto, indices_yellow_in_black;
		color_crop::filter_contours_within(hulls_black, indices_black_proto,
			contours_yellow.first, indices_yellow_in_black);
		hulls_black = color_crop::indices_to_vector(
				hulls_black, indices_black_proto);
		color_crop::filter(color_crop::filter_out_enclosing_contours(
					hulls_black, .51), hulls_black);
		// remove all black hulls within a white hull
		std::vector<int> indices_black, indices_white;
		color_crop::filter_contours_within(hulls_white, indices_white,
				hulls_black, indices_black, .51);
		std::vector<bool> scores_black = color_crop::indices_to_scores(
				indices_black, hulls_black.size());
		color_crop::filter(scores_black, hulls_black, true);
		// concantenate the black and white hulls together
		hulls_white.insert(hulls_white.end(), hulls_black.begin(),
				hulls_black.end());
		// now look for intersecting contours and choose the one with the
		// greater area
		color_crop::filter(color_crop::contour_pair_area_smaller(
					color_crop::contours_find_intersections(hulls_white),
					hulls_white), hulls_white);
		// finally remove enclosing vectors again just in case any all-enclosing
		// black hulls managed to get through
		color_crop::filter(color_crop::filter_out_enclosing_contours(
					hulls_white, .1), hulls_white);
	}


	// sort by area for easier processing
	hulls_white = color_crop::indices_to_vector(hulls_white,
			color_crop::contours_sort_by_area(hulls_white));

	// log the image with contours	
	cv::Mat processedImage;
	processedImage = color_crop::draw_contours(
			hulls_white, img_bgr_enhanced);

	std::vector<cv::Point2f> centroids(hulls_white.size());

	if (masks)
	{
		masks->resize(hulls_white.size());
	} 
	if (hulls)
	{
		*hulls = hulls_white;
	}

	for (int i = 0; i < hulls_white.size(); i++)
	{
		if(masks)
		{
			cv::Mat tmp(img_bgr_enhanced.size(), CV_8UC1,
					cv::Scalar(0));
			color_crop::contour_to_mask(hulls_white[i], tmp);
			(*masks)[i] = tmp;
		}

		centroids[i] = color_crop::contour_centroid(hulls_white[i]);
	}

	return centroids;
}
 
 
int getBins(bool sim, cv::Mat image)
{
	libconfig::Config cfg;
	if(sim) cfg.readFile("dropperSim.conf");
	else cfg.readFile("dropperCom.conf");
	// get the centroids of all bins
 
	//cfg->lookupValue("dropper.depth", depth);
	//cfg->lookupValue("dropper.end_angle_offset", endAngleOffset);
	//cfg->lookupValue("dropper.end_pause_time", endPauseTime);
	//cfg->lookupValue("dropper.image_crop_x", cropx);
	//cfg->lookupValue("dropper.image_crop_y", cropy);
	//cfg->lookupValue("dropper.image_scale_x", scalex);
	//cfg->lookupValue("dropper.image_scale_y", scaley);
//
	// how agressive white balancing should be
	float color_correct_low;
	cfg.lookupValue("dropper.color_correct_low", color_correct_low);
	float color_correct_high;
	cfg.lookupValue("dropper.color_correct_high", color_correct_high);
	// these are the values that are used for the thresholding of each color
	// these should be tweaked as need to attain optimal performance
	//cfg.lookupValue("dropper.yellow_hue_min", yellow_vals[0][0]);
	//cfg.lookupValue("dropper.yellow_hue_max", yellow_vals[0][1]);
	//cfg->lookupValue("dropper.yellow_sat_min", yellow_vals[1][0]);
	//cfg->lookupValue("dropper.yellow_sat_max", yellow_vals[1][1]);
	//cfg->lookupValue("dropper.yellow_val_min", yellow_vals[2][0]);
	//cfg->lookupValue("dropper.yellow_val_max", yellow_vals[2][1]);
	//cfg->lookupValue("dropper.black_hue_min", black_vals[0][0]);
	//cfg->lookupValue("dropper.black_hue_max", black_vals[0][1]);
	//cfg->lookupValue("dropper.black_sat_min", black_vals[1][0]);
	//cfg->lookupValue("dropper.black_sat_max", black_vals[1][1]);
	//cfg->lookupValue("dropper.black_val_min", black_vals[2][0]);
	//cfg->lookupValue("dropper.black_val_max", black_vals[2][1]);
	//cfg->lookupValue("dropper.white_hue_min", white_vals[0][0]);
	//cfg->lookupValue("dropper.white_hue_max", white_vals[0][1]);
	//cfg->lookupValue("dropper.white_sat_min", white_vals[1][0]);
	//cfg->lookupValue("dropper.white_sat_max", white_vals[1][1]);
	//cfg->lookupValue("dropper.white_val_min", white_vals[2][0]);
	//cfg->lookupValue("dropper.white_val_max", white_vals[2][1]);
	//cfg->lookupValue("dropper.orange_hue_min", orange_vals[0][0]);
	//cfg->lookupValue("dropper.orange_hue_max", orange_vals[0][1]);
	//cfg->lookupValue("dropper.orange_sat_min", orange_vals[1][0]);
	//cfg->lookupValue("dropper.orange_sat_max", orange_vals[1][1]);
	//cfg->lookupValue("dropper.orange_val_min", orange_vals[2][0]);
	//cfg->lookupValue("dropper.orange_val_max", orange_vals[2][1]);
	//// min amount of 'redness' for bins to even be considered
	int red_min_integrand;	
	cfg.lookupValue("dropper.red_min_integrand", red_min_integrand);
	//// the min contour areas, contours below which will be ignored
	cfg.lookupValue("dropper.yellow_min_area", yellow_min_area);
	cfg.lookupValue("dropper.black_min_area", black_min_area);
	cfg.lookupValue("dropper.white_min_area", white_min_area);
	cfg.lookupValue("dropper.orange_min_area", orange_min_area);
	cfg.lookupValue("dropper.white_hull_min_area", white_hull_min_area);
	// enhancement might need to turned off for simulator
	bool useenhance;
	cfg.lookupValue("dropper.use_enhance", useenhance);
	// how good subs position over the bin must be before dropping
	//cfg->lookupValue("dropper.min_drop_dist", minDropDist);
	// how long sub should move forward directly prior to dropping to
	// account for the dropper being behind the camera
	//cfg->lookupValue("dropper.offset_time", dropperOffsetTime);
	//// if true it will look for more bins with the black areas directly
	//// if less than the maximum amount of bins are found originally
	cfg.lookupValue("dropper.black_fallback", black_fallback);
	//// how close sub needs to be so that it will not readjust its
	//// heading while approaching the bin
	//cfg->lookupValue("dropper.min_run_dist", minRunDist);
	//// heading to turn to before beginning bins
	//cfg->lookupValue("dropper.angle_to_bins", angle_to_bins);
	//// how long to go forward before looking for bins
	//cfg->lookupValue("dropper.sprint_time", sprint_time);
	//cfg->lookupValue("dropper.timeout", timeout);
 
	if (useenhance)
	{
		// if image does not contain enough high enough value red,
		// it very likely does not contain a bin, and the lack of red
		// would result in a very distorted auto white balance that can
		// lead to false positives, therefore, we assume zero bins
		// immediately.
		cv::Mat channels[image.channels()];
		cv::split(image, channels);
		if (color_crop::integrate(channels[2], 175, 255) < red_min_integrand)
		{
			std::cout<<"found 0 bins \n" << std::endl;
			std::cout<<"bins: \n";
			return 0;
		}

		color_crop::whitebalance_simple_wrapper(image,
				img_enhanced, color_correct_low, color_correct_high);
	}
	else
	{
		img_enhanced = image;
	}
	std::vector<cv::Point2f> centroids = bin_contours(img_enhanced, nullptr, nullptr);

	numBins = 0;
	for (; (numBins < centroids.size()) &&
			(numBins < sizeof(bins)/sizeof(bins[0])); numBins++)
	{
		bins[numBins].x = centroids[numBins].x/img_enhanced.cols - .5f;
		bins[numBins].y = .5f - centroids[numBins].y/img_enhanced.rows;
	}

	std::cout << numBins << "\n";
	for (int i = 0; i < numBins; i++)
	{
		std::cout<< bins[i] << "\n" ;
	}
	std::cout<< std::endl;
	return 0;
}

int main(int argc,char **argv) {
	while(true){
		char temp;
		std::cin.get(temp);
		if(temp == 'd'){
			cv::Mat imagetemp = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
			getBins(*argv[2] == 's',imagetemp);
		}
	}
	return 0;
}
