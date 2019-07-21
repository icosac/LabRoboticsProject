#include <configure.hh>
Filter filter = Filter(30, 30, 30, 100, 100, 100);

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

/*! Function to update trackers with filter*/
void update_trackers(){
  setTrackbarPos("Low H","Filtered Image", filter.low_h);
  setTrackbarPos("Low S","Filtered Image", filter.low_s);
  setTrackbarPos("Low V","Filtered Image", filter.low_v);
  setTrackbarPos("High H","Filtered Image", filter.high_h);
  setTrackbarPos("High S","Filtered Image", filter.high_s);
  setTrackbarPos("High V","Filtered Image", filter.high_v);
}

/*! \brief If DEPLOY is defined then takes a photo from the camera, shows tha various filters and asks if they are
 *  visually correct. If not then it allows to set the various filters through trackbars.
 *  If DEPLOY is not defined then it takes a map from the folder set in Settings and ask for visual confirmation.
 */
void configure (bool deploy, int img_id){
  Settings* s=new Settings();
  s->readFromFile();
  cout << *s << endl;

  Mat frame;
  if (deploy) {
    //Create camera object
    CameraCapture::input_options_t options(1080, 1920, 30, 0);
    CameraCapture *camera = new CameraCapture(options);

    double time;
    if (camera->grab(frame, time)) {
      cout << "Success" << endl;
    } else {
      cout << "Fail getting camera photo." << endl;
      return;
    }
  }
  else {
    cout << "Reading image: " << s->maps(Tuple<int>(1, img_id)).get(0) << endl;
    frame = imread(s->maps(Tuple<int>(1, img_id)).get(0));
  }

#ifdef MY_DEBUG
  my_imshow("prova", frame);
    mywaitkey();
#endif

  cvtColor(frame, frame, COLOR_BGR2HSV);
  if (!show_all_conditions(frame, s)){
    Mat frame_threshold;
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

    //WHITE FILTER
    filter=s->robotMask; update_trackers();
    cout << "Robot filter. " << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      //-- Show the frames
      imshow("Filtered Image",frame_threshold);
    }
    s->changeMask(Settings::WHITE, filter);
    cout << "Robot filter done: " << filter << endl;
  }

  s->writeToFile();

#ifdef DEPLOY
  free(camera);
#endif
  destroyAllWindows();
}

/*! Function to show a picture with various filters taken from Settings. It then asks for visual confirmation.
 *
 * @param frame The image to show.
 * @param s The Settings to use.
 * @return True if the filters are okay, false otherwise.
 */
bool show_all_conditions(const Mat& frame, Settings* s){
  bool ret=false;
  Mat black, red, green, victim, blue, white, robot;

  cout << s->blackMask.Low().val[0] << ", " << s->blackMask.Low().val[1] << ", " << s->blackMask.Low().val[2] << ", " << s->blackMask.High().val[0] << ", " << s->blackMask.High().val[1] << ", " << s->blackMask.High().val[2] << endl;
  cout << s->redMask.Low().val[0] << ", " << s->redMask.Low().val[1] << ", " << s->redMask.Low().val[2] << ", " << s->redMask.High().val[0] << ", " << s->redMask.High().val[1] << ", " << s->redMask.High().val[2] << endl;
  cout << s->greenMask.Low().val[0] << ", " << s->greenMask.Low().val[1] << ", " << s->greenMask.Low().val[2] << ", " << s->greenMask.High().val[0] << ", " << s->greenMask.High().val[1] << ", " << s->greenMask.High().val[2] << endl;
  cout << s->victimMask.Low().val[0] << ", " << s->victimMask.Low().val[1] << ", " << s->victimMask.Low().val[2] << ", " << s->victimMask.High().val[0] << ", " << s->victimMask.High().val[1] << ", " << s->victimMask.High().val[2] << endl;
  cout << s->blueMask.Low().val[0] << ", " << s->blueMask.Low().val[1] << ", " << s->blueMask.Low().val[2] << ", " << s->blueMask.High().val[0] << ", " << s->blueMask.High().val[1] << ", " << s->blueMask.High().val[2] << endl;
  cout << s->whiteMask.Low().val[0] << ", " << s->whiteMask.Low().val[1] << ", " << s->whiteMask.Low().val[2] << ", " << s->whiteMask.High().val[0] << ", " << s->whiteMask.High().val[1] << ", " << s->whiteMask.High().val[2] << endl;
  cout << s->robotMask.Low().val[0] << ", " << s->robotMask.Low().val[1] << ", " << s->robotMask.Low().val[2] << ", " << s->robotMask.High().val[0] << ", " << s->robotMask.High().val[1] << ", " << s->robotMask.High().val[2] << endl;

  inRange(frame, s->blackMask.Low(), s->blackMask.High(), black);
  inRange(frame, s->redMask.Low(), s->redMask.High(), red);
  inRange(frame, s->greenMask.Low(), s->greenMask.High(), green);
  inRange(frame, s->victimMask.Low(), s->victimMask.High(), victim);
  inRange(frame, s->blueMask.Low(), s->blueMask.High(), blue);
  inRange(frame, s->whiteMask.Low(), s->whiteMask.High(), white);
  inRange(frame, s->robotMask.Low(), s->robotMask.High(), robot);

  my_imshow(NAME(frame), frame, true);
  my_imshow(NAME(black), black);
  my_imshow(NAME(red), red);
  my_imshow(NAME(green), green);
  my_imshow(NAME(victim), victim);
  my_imshow(NAME(blue), blue);
  my_imshow(NAME(white), white);
  my_imshow(NAME(robot), robot);

  char c='q';
  cout << "Is this ok? [Y/n]" << endl;
  do{
    c=waitKey(1);
  } while(c!='y' && c!='Y' && c!='n' && c!='N');

  destroyAllWindows();

  if (c=='y' || c=='Y'){
    ret=true;
  }

  return ret;
}