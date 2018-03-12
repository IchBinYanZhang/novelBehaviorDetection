/// this function is develped based on the increemntal clustering algorithm. 
/// feature aggregation step is included with the dynamic clustering algorithm
/// copyright is preserved by Yan Zhang (2017.12.05) ulm university, germany


#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <mutex>
#include <string>
#include <vector>
#include <cmath>
#include <utility> //std::pair

#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <stdio.h>  // snprintf
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/array.hpp>
#include <gflags/gflags.h>

#include <opencv2/core/core.hpp>
// #include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Persistence1D/src/persistence1d/persistence1d.hpp"
#include "mex.h"
// #include "matrix.h"
#include "opencvmex.hpp"


// Global parameters
// int DISPLAY_RESOLUTION_WIDTH;
// int DISPLAY_RESOLUTION_HEIGHT;
// int CAMERA_FRAME_WIDTH;
// int CAMERA_FRAME_HEIGHT;
// int NET_RESOLUTION_WIDTH;
// int NET_RESOLUTION_HEIGHT;
// int BATCH_SIZE;
// float SCALE_GAP;
// float START_SCALE;
// int NUM_GPU;
// std::string PERSON_DETECTOR_CAFFEMODEL; //person detector
// std::string PERSON_DETECTOR_PROTO;      //person detector
// std::string POSE_ESTIMATOR_PROTO;       //pose estimator
// const auto MAX_PEOPLE = RENDER_MAX_PEOPLE;  // defined in render_functions.hpp
// const auto BOX_SIZE = 368;
// const auto BUFFER_SIZE = 4;    //affects latency
// const auto MAX_NUM_PARTS = 70;


enum Dist {Euclidean, Mahalanobis};
      
// global queues for I/O
struct Global {
    Dist dist = Mahalanobis;
    int num_samples = 0;
    int num_dims = 0;
    int VERBOSE = 0;
    float alpha = 1; 
    std::vector<cv::Mat> sample_buffer; // the inputs are stored in std::vector<cv::Mat>
    cv::Mat sample_labels;
    float RADIUS_OFFSET = 0; // initial floating error
    float sigma0;
    struct Codebook {
        int n_clusters;
        cv::Mat n_samples_per_cluster; // N
        cv::Mat cluster_labels; // cluster index
        cv::Mat cluster_locs; // mu
        cv::Mat cluster_stds; // sigma
        cv::Mat cluster_Ex2;  // E[x^2]
    };

    Codebook codebook;
 };



template<typename T>
void test_fun_Cout(T& a){
    std::cout << a << std::endl;
}

struct ColumnCompare
{
    bool operator()(const std::vector<float>& lhs,
                    const std::vector<float>& rhs) const
    {
        return lhs[2] > rhs[2];
        //return lhs[0] > rhs[0];
    }
};

Global global;



float get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time,NULL)) {
        //  Handle error
        return 0;
    }
    return (float)time.tv_sec + (float)time.tv_usec * 1e-6;
    //return (float)time.tv_usec;
}



template<typename T>
class SortingIndexExtractor{
std::vector<T> _vec;
public:
    SortingIndexExtractor(std::vector<T>& vec_in) : _vec(vec_in){}
    bool operator () (const int& a, const int& b) const {
        return _vec[a] < _vec[b];
    }
};



void calClusterRadiusfromStd( std::vector<float>& radius ) {
    for (int i = 0; i < global.codebook.n_clusters; i++) {
        cv::Mat _std = global.codebook.cluster_stds.row(i);
        cv::Mat _var = _std.mul(_std);
    
        float  _var_sum = cv::sum(_var)[0];
        float r_i = (global.alpha * _var_sum>=global.RADIUS_OFFSET)? global.alpha * _var_sum : global.RADIUS_OFFSET;
        radius.push_back(r_i);
    }
}



// void calClusterStd(cv::Mat& samples) {
//     // std::cout<< "[DEBUG] calClusterRadius(cv::Mat& samples) starts..." <<std::endl;
//     global.codebook.n_samples_per_cluster = std::vector<int> (global.codebook.n_clusters, 0); 
//     global.codebook.cluster_std = cv::Mat(global.codebook.cluster_locs.size(), CV_32F);
//     // std::cout<< "[DEBUG] calClusterRadius(cv::Mat& samples) loop starts..." <<std::endl;
//     for (int i = 0; i < global.codebook.n_clusters; i++){
//         cv::Mat _samplei;       

//         for (int j = 0; j < samples.rows; j++) {
//             // std::cout << "[DEBUG] global.codebook.init_sample_labels.at<int>(j) = " << global.codebook.init_sample_labels.at<float>(j) <<std::endl;
//             if (global.codebook.init_sample_labels.at<float>(j)==(float)i){
//                 _samplei.push_back(samples.row(j));    
//                 global.codebook.n_samples_per_cluster[i] +=1;
//             }

//         }
//         // std::cout<< "[DEBUG] calClusterRadius(cv::Mat& samples) global.codebook.n_samples_per_cluster[0] = " 
//         //          << global.codebook.n_samples_per_cluster[0] <<std::endl;
//         int n_samplei = global.codebook.n_samples_per_cluster[i];
//         // std::cout << n_samplei << std::endl;
//         // std::cout<< "[DEBUG] calClusterStd(cv::Mat& samples) _samplei.size() = " << _samplei.size() <<std::endl;
//         cv::Mat _mean = global.codebook.cluster_locs.row(i);
//         cv::Mat _mean_repmat = cv::repeat(_mean, n_samplei, 1);
//         // std::cout << _mean_repmat <<std::endl;
//         cv::Mat _std;
//         cv::reduce ((_samplei-_mean_repmat).mul((_samplei-_mean_repmat))/n_samplei, _std, 0, CV_REDUCE_SUM);
//         // std::cout<< "[DEBUG] calClusterRadius(cv::Mat& samples) after reduce..." <<std::endl;
//         cv::sqrt(_std,_std);
//         _std = _std + global.codebook.init_cluster_std;
//         _std.copyTo(global.codebook.cluster_std.row(i));
        
//     }
//     // std::cout<< "[DEBUG] calClusterRadius(cv::Mat& samples) ends..." <<std::endl;
// }



void showCodebookInfo( ) {
    std::cout << "------------------------Final Result ---------------------------" <<std::endl;
    std::cout << "[CODEBOOK INFO] codebook.n_clusters = " << global.codebook.n_clusters << std::endl;


    // cv::namedWindow("Codebook Image");
    for (int k = 0; k < global.codebook.cluster_locs.rows; k++) {
        std::cout << "[CODEBOOK INFO] cluster - " << k << std::endl;
        std::cout << "[CODEBOOK INFO] codebook.cluster_locs = " << global.codebook.cluster_locs.row(k) << std::endl;
        std::cout << "[CODEBOOK INFO] codebook.cluster_std = " << global.codebook.cluster_stds.row(k)<< std::endl; 
        std::cout << "[CODEBOOK INFO] codebook.num_samples = " << global.codebook.n_samples_per_cluster.row(k)<< std::endl;  
        
    }
}



// Mahalanobis distance -> euclidean distance
void calDistSampleToCluster(cv::Mat& phi, cv::Mat& dist) {
    for(int j = 0; j < global.codebook.cluster_locs.rows; j++) {
        float _dist = 0.0f;

        for (int i = 0; i < phi.cols; i++){
            float x1 = phi.at<float> (i);
            float x2 = global.codebook.cluster_locs.at<float> (j,i);
            
            if (global.dist == Euclidean)
                _dist += (x1-x2)*(x1-x2);
            else
                if(global.dist == Mahalanobis)
                    _dist += (x1-x2)*(x1-x2) / powf(global.codebook.cluster_stds.at<float>(j,i),2);    

        }
        dist.push_back(_dist);
    }
}




// rather than clustering, we only use the first sample
void initializeCodebook() {
    cv::Mat x = global.sample_buffer[0];
    global.codebook.n_clusters = 1;
    global.codebook.n_samples_per_cluster.push_back(1.0f);
    global.codebook.cluster_labels.push_back(0.0f);
    global.codebook.cluster_locs = x;
    global.codebook.cluster_stds = cv::Mat::zeros(1, x.cols, CV_32F);
    global.codebook.cluster_Ex2 = x.mul(x);
    global.sample_labels.push_back(0.0f);
}



void updateCodebook() {

    // start from the second sample to the end
    for(int i = 1; i < global.sample_buffer.size(); i++) {

        cv::Mat _dist, _joints, _joints_repmat;
        double _dist_min;
        int _dist_min_idx; 
        float _radius_min;
        
        std::vector<float> _cluster_radius;
        _joints = global.sample_buffer[i];

        calDistSampleToCluster(_joints, _dist);
        cv::minMaxIdx(_dist, &_dist_min, NULL, &_dist_min_idx);
        
        if (global.dist == Euclidean){
            calClusterRadiusfromStd(_cluster_radius);
            _radius_min = _cluster_radius[_dist_min_idx];
        }else{
            if (global.dist == Mahalanobis)
                _radius_min = (global.sigma0 > 3.0f)? powf(global.sigma0,2) : 9.0f;
        }

        
        
        if (global.VERBOSE)
            std::cout << "[DEBUG] _dist_min = " << _dist_min << "  [DEBUG] _radius_max = " << _radius_min << std::endl;
        
        if ( _dist_min > _radius_min) { // create a new cluster with new location, std=0 and image
            if (global.VERBOSE)
                std::cout<< "[CODEBOOK INFO] updateCodebook(): creating new cluster";

            // new cluster and sample labels
            global.codebook.cluster_labels.push_back((float)global.codebook.n_clusters);
            global.sample_labels.push_back((float)global.codebook.n_clusters);

            // new num_clusters
            global.codebook.n_clusters += 1;

            // new cluster centers
            global.codebook.cluster_locs.push_back(_joints);

            // new cluster stds
            cv::Mat zz = cv::Mat::zeros(1,_joints.cols, CV_32F);
            global.codebook.cluster_stds.push_back(zz);

            // new cluster E[x^2]
            cv::Mat ex2 = _joints.mul(_joints);
            global.codebook.cluster_Ex2.push_back(ex2);

            // number of samples in the new clusters
            global.codebook.n_samples_per_cluster.push_back(1.0f);



        }
        else {  

            if (global.VERBOSE)
                std::cout<< "[CODEBOOK INFO] updateCodebook(): updating current cluster " << _dist_min_idx << std::endl;

            // assign a label to the current sample
            global.sample_labels.push_back((float)_dist_min_idx);

            float _n_samplei = global.codebook.n_samples_per_cluster.at<float>(_dist_min_idx);
            cv::Mat _mean = global.codebook.cluster_locs.row(_dist_min_idx).clone();
            cv::Mat _std = global.codebook.cluster_stds.row(_dist_min_idx).clone();
            cv::Mat _Ex2 = global.codebook.cluster_Ex2.row(_dist_min_idx).clone();

            // update cluster centers 
            global.codebook.cluster_locs.row(_dist_min_idx) = _mean + (_joints-_mean ) / (float) (1+_n_samplei);

            // updating Ex2
            global.codebook.cluster_Ex2.row(_dist_min_idx) = (_Ex2*_n_samplei + _joints.mul(_joints))/(float)(_n_samplei+1);

            // updating std
            _std = global.codebook.cluster_Ex2.row(_dist_min_idx) 
                  - global.codebook.cluster_locs.row(_dist_min_idx).mul(global.codebook.cluster_locs.row(_dist_min_idx));

            cv::sqrt(_std, global.codebook.cluster_stds.row(_dist_min_idx));        

            // update sample numbers
            global.codebook.n_samples_per_cluster.at<float>(_dist_min_idx) = _n_samplei + 1.0f;
        }

    }

}











// the dynamic EM algorithm does not require clustering for initialization
void runDynamicEM() {
    updateCodebook( );
}





// void assignLabelsToSamples(const cv::Mat& samples, cv::Mat& labels){
//     double _dist_min;
//     int _dist_min_idx;

//     for (int i = 0; i < global.num_samples; i++){
//         cv::Mat phi = samples.row(i);
//         cv::Mat dist;

//         calDistSampleToCluster(phi, dist);
//         cv::minMaxIdx(dist, &_dist_min, NULL, &_dist_min_idx);
//         labels.push_back(_dist_min_idx);
//     }
// }



// initialize the codebook based on the inputs
void parseArgsToCodebook(const cv::Mat& cluster_locs0, const cv::Mat& cluster_stds0,
                         const cv::Mat& cluster_Ex20, const cv::Mat& cluster_sizes0){
    global.codebook.n_clusters = cluster_locs0.rows;
    for(int i = 0; i < cluster_locs0.rows; i++){
        global.codebook.n_samples_per_cluster.push_back(cluster_sizes0.at<float>(i));
        global.codebook.cluster_labels.at<float>(i) = float(i);
    }

    global.codebook.cluster_locs = cluster_locs0.clone();
    global.codebook.cluster_stds = cluster_stds0.clone();
    global.codebook.cluster_Ex2 = cluster_Ex20.clone();
}




// Input: all the pose patterns. Rows - samples, cols - features
// Output: codebook, segment boudnaries.
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 8 && nrhs != 4)
        mexErrMsgIdAndTxt("Three arguments should be parsed", 
            "usage: dynamicEM(samples, sigma_0, dist_type, verbose, (cluster_locs, cluster_stds, cluster_Ex2, cluster_sizes)");

    
    // parse arguments
    global.sigma0 = (float)*mxGetPr(prhs[1]);
    int dist_type =  (int)*mxGetPr(prhs[2]);
    if (dist_type==0)
        global.dist = Euclidean;
    else
        if(dist_type == 1)
            global.dist = Mahalanobis;


    global.VERBOSE = (int)*mxGetPr(prhs[3]);
    
    cv::Mat all_samples;
    ocvMxArrayToMat_double(prhs[0], all_samples);
    all_samples.convertTo(all_samples, CV_32F);
    
    global.num_samples = all_samples.rows;
    global.num_dims = all_samples.cols;
    global.RADIUS_OFFSET = global.sigma0*(global.num_dims); // sigma0*dimensions
    global.sample_buffer.clear();

    for(int i = 0; i < global.num_samples; i++)
        global.sample_buffer.push_back(all_samples.row(i));
        

    global.sample_labels.release();
    if (nrhs == 8){
        cv::Mat cluster_sizes0, cluster_locs0, cluster_stds0, cluster_Ex20;
        ocvMxArrayToMat_double(prhs[4], cluster_locs0);
        cluster_locs0.convertTo(cluster_locs0, CV_32F);
        ocvMxArrayToMat_double(prhs[5], cluster_stds0);
        cluster_stds0.convertTo(cluster_stds0, CV_32F);
        ocvMxArrayToMat_double(prhs[6], cluster_Ex20);
        cluster_Ex20.convertTo(cluster_Ex20, CV_32F);
        ocvMxArrayToMat_double(prhs[7], cluster_sizes0);
        cluster_sizes0.convertTo(cluster_sizes0, CV_32F);

        parseArgsToCodebook(cluster_locs0, cluster_stds0, cluster_Ex20, cluster_sizes0);
    }else{
        global.codebook.cluster_labels.release();
        global.codebook.cluster_locs.release();
        global.codebook.cluster_stds.release();
        global.codebook.n_samples_per_cluster.release();
        initializeCodebook();
    }

    runDynamicEM();
    if(global.VERBOSE)
        showCodebookInfo();

    cv::Mat labels;
    cv::Mat cluster_locs;
    cv::Mat cluster_stds;
    cv::Mat cluster_Ex2;
    cv::Mat cluster_sizes;
    global.sample_labels.convertTo(labels, CV_64F);
    global.codebook.cluster_locs.convertTo(cluster_locs, CV_64F);
    global.codebook.cluster_stds.convertTo(cluster_stds, CV_64F);
    global.codebook.cluster_Ex2.convertTo(cluster_Ex2, CV_64F);
    global.codebook.n_samples_per_cluster.convertTo(cluster_sizes, CV_64F);


    plhs[0] = ocvMxArrayFromMat_double(labels);
    plhs[1] = ocvMxArrayFromMat_double(cluster_locs);
    plhs[2] = ocvMxArrayFromMat_double(cluster_stds);
    plhs[3] = ocvMxArrayFromMat_double(cluster_Ex2);
    plhs[4] = ocvMxArrayFromMat_double(cluster_sizes);

}


