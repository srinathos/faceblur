/**
 * main.cpp
 *
 * ---- Application in development ----
 *
 * This program detects faces in a video and blurs them.
 * Currently using the dlib CNN based face detection API.
 *
 * Code adapted from:
 *          http://dlib.net/dnn_mmod_face_detection_ex.cpp.html
 *
 * Working:
 *          - Face detection
 *          - Face blurring
 *
 * To do
 *          - Fix terrible speed,
 *              - need to compare performance with Haar cascades
 *          - Deal with multiple resolutions
 *              -pyramid up if low, downscale if high
 * **/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "dlib/dnn.h"
#include "dlib/opencv.h"

using namespace std;
using namespace dlib;
using namespace cv;


// Defining the CNN structure

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

static bool DEBUG = true;

void usage(){
    cout << "Usage:" << endl;
    cout << "./faceblur mmod_human_face_detector.dat <video_file>" << endl;
    cout << "\nmmod_human_face_detector: Pretrained CNN model";
    cout << "\n\nYou can get the mmod_human_face_detector.dat file from:\n";
    cout << "http://dlib.net/files/mmod_human_face_detector.dat.bz2" << endl;
}


int main(int argc, char** argv){
    if (argc < 3){
        usage();
        return 0;
    }

    net_type net;

    // loading model params
    deserialize(argv[1]) >> net;

    // test value to downscale by
    float scale_factor = 0.7f;

    // Loading input video...
    VideoCapture input_video(argv[2]);
    if(!input_video.isOpened()) {
        cout << "Error opening input file.\n Exiting...";
        return -1;
    }

    Mat input_frame, resized_input;

    // Grabbing sample frame and info to determine output_file
    input_video >> input_frame;
    auto ex = static_cast<int>(input_video.get(CV_CAP_PROP_FOURCC));
    // Calculating output video size
    Size S = Size((int) ceil(scale_factor * input_video.get(CV_CAP_PROP_FRAME_WIDTH)),
                  (int) ceil(scale_factor * input_video.get(CV_CAP_PROP_FRAME_HEIGHT)));

    // Initializing video out stream
    VideoWriter output_video("../out/output_test.mp4", ex, input_video.get(CV_CAP_PROP_FPS), S, true);

    if (!output_video.isOpened())
    {
        cout  << "Could not open the output video for write";
        return -1;
    }

    // Loop for every frame in the video
    for(int i = 0;;i++)
    {
        // Skip every 4th frame, test only
        if(i % 4 != 0)
            continue;

        // grabbing frame, resizing and converting to appropriate input for dlib's CNN
        input_video >> input_frame;
        resize(input_frame, resized_input, Size(), scale_factor, scale_factor);
        matrix<rgb_pixel> cnn_input;
        assign_image(cnn_input, cv_image<bgr_pixel>(resized_input));

        // Upsampling
//        while(frame.size() < 1800*1800)
//            pyramid_up(frame);

        // Detecting faces
        auto dets = net(cnn_input);

        if (DEBUG){
            cout << "Number of faces detected: " << dets.size() << endl;
        }

        for (auto&& d : dets) {
            // Grabbing region of interest and applying gaussian blur
            cv::Rect roi(Point2f(d.rect.left(),d.rect.top()), Point2f(d.rect.right(),d.rect.bottom()));
            cv::GaussianBlur(resized_input(roi), resized_input(roi), Size(0,0), 10);
        }
        if (DEBUG){
            cout << "Number of faces detected: " << dets.size() << endl;
            imshow("Output frame",resized_input);
        }

        output_video << resized_input;
        if(waitKey(30) >= 0) break;
    }
    return 0;
}



