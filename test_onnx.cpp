#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;
#define _DEBUG

/// @brief  A utility function to print out the elements of vector
template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> vec) {
	os << "[ ";
	std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, " "));
	os << " ]";
	return os;
}


bool test_onnx(string modelFileName, vector<int> inputShape);

int main() {
	std::pair<string, vector<int>> attr("attr_b2.onnx",
		vector<int>{ 2, 3, 256, 128});
	std::pair<string, vector<int>> pose("pose.onnx",
		vector<int>{ 5, 3, 192, 256 });
	std::pair<string, vector<int>> act("act_b8.onnx",
		vector<int>{ 8, 1, 17, 48, 64, 64 });

	vector <std::pair<string, vector<int>>> fileWithInputShape{ attr, pose, act };
	vector<std::pair<string, bool>> testResults;

	for (const auto& pair : fileWithInputShape) {
		string prefix = "C:/Users/ha/Desktop/test_onnx/";
		string fileOnxx = prefix + pair.first;
		vector<int> inputShape = pair.second;
		cout << endl << "***********************************************" << endl;
		cout << "***TEST -" << fileOnxx << "-********" << endl;
		bool modelResult = test_onnx(fileOnxx, inputShape);
		testResults.push_back(std::make_pair(fileOnxx, modelResult));
	}

	cout << endl << "***************FINAL RESULTS********************";
	for (const auto& test : testResults) {
		string model = test.first;
		string result = test.second ? "Succeeded" : "Failed";
		cout << endl << model << ":\t" << result;
	}
	cout << endl << "***************END RESULTS********************" << endl;

	return 0;
};

bool test_onnx(string fileOnnx, vector<int> inputShape) {
	//cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
	try {
		dnn::Net net;
		try {
			net = dnn::readNet(fileOnnx);
			cout << "readNet(" << fileOnnx << ")" << endl;

		}
		catch (cv::Exception e) {
			cout << "Load ONNX failed" << endl;
			return false;
		}
		net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
		Mat blobInput = Mat::ones(inputShape.size(), inputShape.data(), CV_32F);
		cout << endl << "---------------------------------" << endl;
		cout << "blobInput(" << blobInput.size << ")" << endl;
		net.setInput(blobInput);
		Mat out = net.forward();
		cout << "netOut1st(" << out.size << ")" << endl;
		cout << "---------------------------------" << endl;

		return true;

	}
	catch (cv::Exception e) {
		cout << "ERROR:" << e.msg << endl;
		return false;
	}
}
