#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::face;
namespace fs = std::filesystem;

class FaceRecognition {
private:
    Ptr<LBPHFaceRecognizer> recognizer;
    string modelPath = "trained_model.yml"; // Path to save/load model

public:
    FaceRecognition() {
        recognizer = LBPHFaceRecognizer::create();
    }

    //Preprocessing
    Mat preprocessImage(const Mat& input) {
        Mat gray, resized;

        cvtColor(input, gray, COLOR_BGR2GRAY);

        resize(gray, resized, Size(320, 240));

        return resized;
    }

    //Dataset loading
    void loadDataset(const string& path, vector<Mat>& images, vector<int>& labels) {
        int label = 0;
        for (const auto& dirEntry : fs::directory_iterator(path)) {
            if (dirEntry.is_directory()) {
                for (const auto& file : fs::directory_iterator(dirEntry.path())) {
                    Mat img = imread(file.path().string());
                    if (!img.empty()) {
                        images.push_back(preprocessImage(img));
                        labels.push_back(label);
                    }
                }
                label++;
            }
        }
    }

    // Function to train the model
    void trainModel(const string& datasetPath) {
        vector<Mat> images;
        vector<int> labels;

        loadDataset(datasetPath, images, labels);

        recognizer->train(images, labels);

        recognizer->save(modelPath);

        cout << "Model trained and saved to " << modelPath << endl;
    }

    void loadModel() {
        try {
            recognizer->read(modelPath);
            cout << "Model loaded successfully!" << endl;
        }
        catch (const cv::Exception& e) {
            cout << "Error loading model! " << endl;
        }
    }

    void recognizeFace(const Mat& inputImage) {
        Mat preprocessedImage = preprocessImage(inputImage);

        int predictedLabel;
        double confidence;

        recognizer->predict(preprocessedImage, predictedLabel, confidence);

        if (confidence < 90) { // Confidence threshold for image detection (adjustable)
            cout << "Predicted label: " << predictedLabel << " with confidence: " << confidence << endl;
        }
        else {
            cout << "Unknown face!" << endl;
        }
    }


    void recognizeFromWebcam() {
        VideoCapture cap(1); // 0 for real webcam, 1 for OBS VirtualCam
        if (!cap.isOpened()) {
            cerr << "Error: Could not open webcam!" << endl;
            return;
        }

        Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            // Detect faces in the frame (using Haar cascade)
            CascadeClassifier faceCascade;
            if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
                cerr << "Error: Could not load Haar cascade!" << endl;
                break;
            }

            vector<Rect> faces;
            faceCascade.detectMultiScale(frame, faces);

            for (const auto& face : faces) {
                Mat faceRegion = frame(face);

                int predictedLabel = -1;
                double confidence = 0.0;

                Mat preprocessedFace = preprocessImage(faceRegion);
                recognizer->predict(preprocessedFace, predictedLabel, confidence);

                string labelText;
                if (confidence < 90) { // Confidence threshold for webcam detection (adjustable)
                    labelText = "Label: " + to_string(predictedLabel) +
                        " (" + to_string(int(confidence)) + ")";
                }
                else {
                    labelText = "Unknown";
                }

                rectangle(frame, face, Scalar(255, 0, 0), 2);

                // Put the label text above the rectangle
                int baseline = 0;
                Size textSize = getTextSize(labelText, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                Point textOrg(face.x, face.y - 5); // Place text slightly above the rectangle

                putText(frame, labelText, textOrg, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            }

            imshow("Face Recognition", frame);

            if (waitKey(10) == 'q') {
                break;
            }
        }
    }

};

int main() {
    FaceRecognition faceRec;

    cout << "1. Train the model\n2. Recognize from webcam\n3. Recognize from image\n";
    int choice;
    cin >> choice;

    switch (choice)
    {
    case(1): {
        //Training the face recognition data model by providing a dataset
        string datasetPath;
        cout << "Enter the dataset folder path: ";
        cin >> datasetPath;
        faceRec.trainModel(datasetPath);
        break;
    }
    case(2): {
        //Real-time face recognition with webcam
        faceRec.loadModel();
        faceRec.recognizeFromWebcam();
        break;
    }
    case(3): {
        //Picture based face recognition using trained model
        faceRec.loadModel();
        string imagePath;
        cout << "Enter the image path: ";
        cin >> imagePath;
        Mat img = imread(imagePath);
        faceRec.recognizeFace(img);
        break;
    }
    default:
        cout << "Invalid option!" << endl;
    }

    return 0;
}