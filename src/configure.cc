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

/*! \brief It acqire a frame from the default camera of the pc.

    \param[in] save If save, or not, the acquired image to a file.
    \return The Mat of the acquired frame.
*/
Mat acquireImage(const bool save){
  cout << "I get in" << endl;
  Mat frame;
  //Create camera object
  CameraCapture::input_options_t options(1080, 1920, 30, 0);
  CameraCapture *camera = new CameraCapture(options);
  
  double time;
  if (camera->grab(frame, time)) {
    #ifdef DEBUG
      cout << "Frame grabbed successfully" << endl;
    #endif
  } else {
    throw MyException<string>(EXCEPTION_TYPE::GENERAL, "Fail getting camera photo.", __LINE__, __FILE__);    
  }
  
  if(save){
    imwrite("data/map/01.jpg", frame);
  }
  delete camera;
  return(frame);
}

/*! \brief If DEPLOY is defined then takes a photo from the camera, shows tha various filters and asks if they are
 *  visually correct. If not then it allows to set the various filters through trackbars.
 *  If DEPLOY is not defined then it takes a map from the folder set in Settings and ask for visual confirmation.
 */
void configure (bool deploy, int img_id){

  Mat frame;
  if (deploy) {
    frame = acquireImage(true);
  }
  else {
    frame = imread(sett->maps(Tuple<int>(1, img_id)).get(0));
  }

#ifdef DEBUG
  my_imshow("Frame RGB", frame);
  mywaitkey();
#endif

  cvtColor(frame, frame, COLOR_BGR2HSV);
  if (!show_all_conditions(frame)){
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
    filter=sett->blackMask; update_trackers();
    cout << "Black filter. " << filter << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      imshow("Filtered Image", frame_threshold);
    }
    sett->changeMask(Settings::BLACK, filter);
    cout << "Black filter done: " << filter << endl;

    //RED FILTER
    filter=sett->redMask; update_trackers();
    cout << "Red filter. " << filter << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      imshow("Filtered Image",frame_threshold);
    }
    sett->changeMask(Settings::RED, filter);
    cout << "Red filter done: " << filter << endl;

    //GREEN FILTER
    filter=sett->greenMask; update_trackers();
    cout << "Green filter. " << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      imshow("Filtered Image",frame_threshold);
    }
    sett->changeMask(Settings::GREEN, filter);
    cout << "Green filter done: " << filter << endl;

    //VICTIMS FILTER
    filter=sett->victimMask; update_trackers();
    cout << "Victim filter. " << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      imshow("Filtered Image",frame_threshold);
    }
    sett->changeMask(Settings::VICTIMS, filter);
    cout << "Victims filter done: " << filter << endl;

    //BLUE FILTER
    filter=sett->blueMask; update_trackers();
    cout << "Blue filter. " << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      imshow("Filtered Image",frame_threshold);
    }
    sett->changeMask(Settings::BLUE, filter);
    cout << "Gate blue done: " << filter << endl;

    //WHITE FILTER
    // filter=sett->whiteMask; update_trackers();
    // cout << "White filter. " << endl;
    // while ((char)waitKey(1)!='c'){
    //   inRange(frame, filter.Low(), filter.High(), frame_threshold);
    //   imshow("Filtered Image",frame_threshold);
    // }
    // sett->changeMask(Settings::WHITE, filter);
    // cout << "White filter done: " << filter << endl;

    //ROBOT FILTER
    filter=sett->robotMask; update_trackers();
    cout << "Robot filter. " << endl;
    while ((char)waitKey(1)!='c'){
      inRange(frame, filter.Low(), filter.High(), frame_threshold);
      imshow("Filtered Image",frame_threshold);
    }
    sett->changeMask(Settings::ROBOT, filter);
    cout << "Robot filter done: " << filter << endl;
  }

  sett->save();
  sett->writeToFile();

  cout << *sett << endl;
  
  destroyAllWindows();
}

/*! Function to show a picture with various filters taken from Settings. It then asks for visual confirmation.
 *
 * @param frame The image to show.
 * @return True if the filters are okay, false otherwise.
 */
bool show_all_conditions(const Mat& frame){
  bool ret=false;
  Mat black, red, green, victim, blue, white, robot;

  inRange(frame, sett->blackMask.Low(), sett->blackMask.High(), black);
  inRange(frame, sett->redMask.Low(), sett->redMask.High(), red);
  inRange(frame, sett->greenMask.Low(), sett->greenMask.High(), green);
  inRange(frame, sett->victimMask.Low(), sett->victimMask.High(), victim);
  inRange(frame, sett->blueMask.Low(), sett->blueMask.High(), blue);
  // inRange(frame, sett->whiteMask.Low(), sett->whiteMask.High(), white);
  inRange(frame, sett->robotMask.Low(), sett->robotMask.High(), robot);

  my_imshow(NAME(frame), frame, true);
  my_imshow(NAME(black), black);
  my_imshow(NAME(red), red);
  my_imshow(NAME(green), green);
  my_imshow(NAME(victim), victim);
  my_imshow(NAME(blue), blue);
  // my_imshow(NAME(white), white);
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
