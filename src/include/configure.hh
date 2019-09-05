#ifndef CONFIGURE_HH
#define CONFIGURE_HH

#include<iostream>

#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core/core_c.h>

#include<utils.hh>
#include<filter.hh>
#include<camera_capture.hh>
#include<settings.hh>

using namespace std;
using namespace cv;

extern CameraCapture* camera;

/*! \brief It acqire a frame from the default camera of the pc.

    \param[in] save If save, or not, the acquired image to a file.
    \return The Mat of the acquired frame.
*/
Mat acquireImage(const bool save=true);

/*! \brief If deploy is true then takes a photo from the camera, shows tha various filters and asks if they are
 *  visually correct. If not then it allows to set the various filters through trackbars.
 *  If deploy is false then it takes the imd_id-th maps from the folder set in Settings and ask for visual confirmation.
 */
void configure(bool deploy=true, int img_id=0);

/*! Function to show a picture with various filters taken from Settings. It then asks for visual confirmation.
 *
 * @param frame The image to show.
 * @return True if the filters are okay, false otherwise.
 */
bool show_all_conditions(const Mat& frame);

#endif
