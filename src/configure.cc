#include <configure.hh>
Filter filter = Filter(30, 30, 30, 100, 100, 100);

void configure (){
  Settings* s=new Settings();
  s->readFromFile();

  Mat frame;
#ifdef DEPLOY
  //Create camera object
	CameraCapture::input_options_t options(1080, 1920, 30, 0);
	CameraCapture *camera=new CameraCapture(options);

	double time;
	if (camera->grab(frame, time)){
		cout << "Success" << endl;
	}
	else {
		cout << "Fail getting camera photo." << endl;
		return 0;
	}
#else
  cout << "Reading image: " << s->maps(0).get(0) << endl;
  frame=imread(s->maps(0).get(0));
//  frame=cvLoadImageM(s->maps(0).get(0).c_str());
#endif

#ifdef MY_DEBUG
  my_imshow("prova", frame);
    mywaitkey();
#endif

  if (show_all_conditions(frame, s)){
    Mat frame_threshold;
    cvtColor(frame, frame, COLOR_BGR2HSV);
    namedWindow("Filtered Image", WINDOW_NORMAL);

    //-- Trackbars to set thresholds for RGB values
    createTrackbar("Low H","Filtered Image", &(filter.low_h), 180, on_low_h_thresh_trackbar);
    createTrackbar("Low S","Filtered Image", &(filter.low_s), 255, on_low_s_thresh_trackbar);
    createTrackbar("Low V","Filtered Image", &(filter.low_v), 255, on_low_v_thresh_trackbar);
    createTrackbar("High H","Filtered Image", &(filter.high_h), 180, on_high_h_thresh_trackbar);
    createTrackbar("High S","Filtered Image", &(filter.high_s), 255, on_high_s_thresh_trackbar);
    createTrackbar("High V","Filtered Image", &(filter.high_v), 255, on_high_v_thresh_trackbar);

    //BLACK FILTER
    filter=s->blackMask; update_trackers();
    cout << "Black filter. " << filter << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      //-- Show the frames
      imshow("Filtered Image", frame_threshold);
    }
    s->changeMask(Settings::BLACK, filter);
    cout << "Black filter done: " << filter << endl;

    //RED FILTER
    filter=s->redMask; update_trackers();
    cout << "Red filter. " << filter << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      //-- Show the frames
      imshow("Filtered Image",frame_threshold);
    }
    s->changeMask(Settings::RED, filter);
    cout << "Red filter done: " << filter << endl;

    //GREEN FILTER
    filter=s->greenMask; update_trackers();
    cout << "Green filter. " << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      //-- Show the frames
      imshow("Filtered Image",frame_threshold);
    }
    s->changeMask(Settings::GREEN, filter);
    cout << "Green filter done: " << filter << endl;

    //VICTIMS FILTER
    filter=s->victimMask; update_trackers();
    cout << "Victim filter. " << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      //-- Show the frames
      imshow("Filtered Image",frame_threshold);
    }
    s->changeMask(Settings::VICTIMS, filter);
    cout << "Victims filter done: " << filter << endl;

    //BLUE FILTER
    filter=s->blueMask; update_trackers();
    cout << "Blue filter. " << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      //-- Show the frames
      imshow("Filtered Image",frame_threshold);
    }
    s->changeMask(Settings::BLUE, filter);
    cout << "Victims blue done: " << filter << endl;

    //WHITE FILTER
    filter=s->whiteMask; update_trackers();
    cout << "White filter. " << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      //-- Show the frames
      imshow("Filtered Image",frame_threshold);
    }
    s->changeMask(Settings::WHITE, filter);
    cout << "White filter done: " << filter << endl;
  }

  s->writeToFile();

#ifdef DEPLOY
  free(camera);
#endif
}

bool show_all_conditions(const Mat& frame, Settings* s){
  bool ret=true;
  Mat black, red, green, victim, blue, white;

  inRange(frame, s->blackMask.Low(), s->blackMask.High(), black);
  inRange(frame, s->redMask.Low(), s->redMask.High(), red);
  inRange(frame, s->greenMask.Low(), s->greenMask.High(), green);
  inRange(frame, s->victimMask.Low(), s->victimMask.High(), victim);
  inRange(frame, s->blueMask.Low(), s->blueMask.High(), blue);
  inRange(frame, s->whiteMask.Low(), s->whiteMask.High(), white);

  my_imshow(NAME(frame), frame, true);
  my_imshow(NAME(black), black);
  my_imshow(NAME(red), red);
  my_imshow(NAME(green), green);
  my_imshow(NAME(victim), victim);
  my_imshow(NAME(blue), blue);
  my_imshow(NAME(white), white);

  char c='q';
  cout << "Is this ok? [Y/n]" << endl;
  do{
    c=waitKey(1);
  } while(c!='y' && c!='Y' && c!='n' && c!='N');

  destroyAllWindows();

  if (c=='y' || c=='Y'){
    ret=false;
  }

  return ret;
}

void update_trackers(){
  setTrackbarPos("Low H","Filtered Image", filter.low_h);
  setTrackbarPos("Low S","Filtered Image", filter.low_s);
  setTrackbarPos("Low V","Filtered Image", filter.low_v);
  setTrackbarPos("High H","Filtered Image", filter.high_h);
  setTrackbarPos("High S","Filtered Image", filter.high_s);
  setTrackbarPos("High V","Filtered Image", filter.high_v);
}

/** @function on_low_h_thresh_trackbar */
void on_low_h_thresh_trackbar(int, void *)
{
  filter.low_h = min(filter.high_h-1, filter.low_h);
  setTrackbarPos("Low H","Filtered Image", filter.low_h);
}

/** @function on_high_h_thresh_trackbar */
void on_high_h_thresh_trackbar(int, void *)
{
  filter.high_h = max(filter.high_h, filter.low_h+1);
  setTrackbarPos("High H", "Filtered Image", filter.high_h);
}

/** @function on_low_s_thresh_trackbar */
void on_low_s_thresh_trackbar(int, void *)
{
  filter.low_s = min(filter.high_s-1, filter.low_s);
  setTrackbarPos("Low S","Filtered Image", filter.low_s);
}

/** @function on_high_s_thresh_trackbar */
void on_high_s_thresh_trackbar(int, void *)
{
  filter.high_s = max(filter.high_s, filter.low_s+1);
  setTrackbarPos("High S", "Filtered Image", filter.high_s);
}

/** @function on_low_v_thresh_trackbar */
void on_low_v_thresh_trackbar(int, void *)
{
  filter.low_v= min(filter.high_v-1, filter.low_v);
  setTrackbarPos("Low V","Filtered Image", filter.low_v);
}

/** @function on_high_v_thresh_trackbar */
void on_high_v_thresh_trackbar(int, void *)
{
  filter.high_v = max(filter.high_v, filter.low_v+1);
  setTrackbarPos("High V", "Filtered Image", filter.high_v);
}
