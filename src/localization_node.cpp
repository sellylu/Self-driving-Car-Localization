/**
 * @file localization_node.cpp
 *
 * @brief real-time localize self-driving car location accroding to LiDAR sensor point cloud
 *
 * @author Selly Lu
 *
 */
#include <omp.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

#include <ros/ros.h>
#include <ros/package.h>

#include <sensor_msgs/PointCloud2.h>
#include "sensor_msgs/Imu.h"
#include <visualization_msgs/Marker.h>

#include <Eigen/Dense>
#include "eigen_conversions/eigen_msg.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/transforms.h>


#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

using namespace std;

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr filter_cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr filter_scan_cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);

pcl::PCLPointCloud2::Ptr cloud_pcl(new pcl::PCLPointCloud2());
pcl::PCLPointCloud2::Ptr filter_cloud_pcl(new pcl::PCLPointCloud2());
pcl::PCLPointCloud2::Ptr crop_cloud_pcl(new pcl::PCLPointCloud2());
pcl::PCLPointCloud2::Ptr scan_cloud_pcl(new pcl::PCLPointCloud2());
pcl::PCLPointCloud2::Ptr filter_scan_cloud_pcl(new pcl::PCLPointCloud2());
pcl::PCLPointCloud2::Ptr crop_scan_cloud_pcl(new pcl::PCLPointCloud2());

sensor_msgs::PointCloud2::Ptr cloud_ros(new sensor_msgs::PointCloud2);
sensor_msgs::PointCloud2::Ptr filter_cloud_ros(new sensor_msgs::PointCloud2);

ros::Publisher map_pub,scan_pub,map_filtered_pub,scan_filtered_pub, marker_pub;
ros::Subscriber scan_sub, point_sub, imu_sub;

pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();

clock_t t;
fstream file;
int file_num;
string result_path;
int sec, nsec;
float prev_score = 0;

bool angle_init = false;
bool calibrate_pos = false;
bool start_output = false;
vector<geometry_msgs::Point> gps_buffer;
visualization_msgs::Marker line_strip;

void filtering(pcl::PCLPointCloud2::Ptr input_cloud_pcl,
               pcl::PCLPointCloud2::Ptr filter_cloud_pcl,
               float leaf_size) {
    pcl::VoxelGrid<pcl::PCLPointCloud2> filter_voxel;
    filter_voxel.setInputCloud (input_cloud_pcl);
    filter_voxel.setLeafSize (leaf_size,leaf_size,leaf_size);
    filter_voxel.filter (*filter_cloud_pcl);
}

void submap(pcl::PCLPointCloud2::Ptr input_cloud_pcl,
            pcl::PCLPointCloud2::Ptr filter_cloud_pcl,
            bool origin,
            int removeGround) {
    geometry_msgs::Point now_gps = gps_buffer.back();
    float x=0,y=0,z=0;
    if(!origin) {
        x=guess(0,3);
        y=guess(1,3);
        z=guess(2,3);
    }
    int range = 100;
    z -= removeGround;
    pcl::CropBox<pcl::PCLPointCloud2> filter;
    filter.setInputCloud(input_cloud_pcl);
    filter.setMin(Eigen::Vector4f(x-range,y-range,z,1));
    filter.setMax(Eigen::Vector4f(x+range,y+range,z+range,1));
    filter.filter(*filter_cloud_pcl);
}

void removeGround(pcl::PointCloud<pcl::PointXYZ>::Ptr filter_scan_cloud_xyz, 
                  pcl::PointIndices::Ptr indicesptr) {
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    // seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.1);//original is 0.05 ->0.3

    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (filter_scan_cloud_xyz);
    seg.segment (*inliers, *coefficients);
    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (filter_scan_cloud_xyz);
    extract.setIndices (inliers);
    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*filter_scan_cloud_xyz);
}

void denoise(pcl::PointCloud<pcl::PointXYZ>::Ptr filter_scan_cloud_xyz) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (filter_scan_cloud_xyz);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*filter_scan_cloud_xyz);
}

void calibratePosition(int i, Eigen::Matrix4f tmp, 
                       vector<float> &score, vector<Eigen::Matrix4f> &trans,
                       pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_func,
                       int tol) {
    // shift x
    if(i>=3 && i<=5) {
        tmp(0,3) -= tol;
    } else if (i<=1 && i>= 7){
        tmp(0,3) += tol;
    }

    // shift y
    if(i>=1 && i<=3){
        tmp(1,3) += tol;
    } else if(i>=5 && i<=7){
        tmp(1,3) -= tol;
    }
    pcl::PointCloud<pcl::PointXYZ> final_cloud_xyz;

    icp_func.align(final_cloud_xyz, tmp);

    score[i] = (icp_func.getFitnessScore());
    trans[i] = (icp_func.getFinalTransformation());
}

void scanCallback(sensor_msgs::PointCloud2::ConstPtr scan_msg) {
    if (!angle_init)
        return;

    pcl_conversions::toPCL(*scan_msg, *scan_cloud_pcl);
    // pcl::fromPCLPointCloud2(*scan_cloud_pcl, *scan_cloud_xyz);
    // downsample by voxelgrid
    filtering(scan_cloud_pcl, filter_scan_cloud_pcl,0.2);
    // std::cerr<<"The points data:  "<<scan_cloud_pcl->data.size()<<std::endl;
    // if (prev_score > 0.1) {
    //  submap(scan_cloud_pcl,filter_scan_cloud_pcl, true, 1);
    // } else {
        submap(filter_scan_cloud_pcl,crop_scan_cloud_pcl, true, 1);
    // }
    // std::cerr<<"The points data:  "<<filter_scan_cloud_pcl->data.size()<<std::endl;
    pcl::fromPCLPointCloud2(*crop_scan_cloud_pcl, *filter_scan_cloud_xyz);



    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    pcl::PointXYZ origin(0.0f, 0.0f, 0.0f);
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(filter_scan_cloud_xyz);
    kdtree.radiusSearch(origin, 15, pointIdxRadiusSearch, pointRadiusSquaredDistance);

    boost::shared_ptr<std::vector<int>> indicesptr (new std::vector<int> (pointIdxRadiusSearch)); // Convert to Point Indices
    pcl::ExtractIndices<pcl::PointXYZ> indiceFilter(true);
    indiceFilter.setInputCloud(filter_scan_cloud_xyz);
    indiceFilter.setIndices(indicesptr);
    indiceFilter.setKeepOrganized(false);
    pcl::PointCloud<pcl::PointXYZ>::Ptr close_scan_cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);;
    indiceFilter.filter(*close_scan_cloud_xyz);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    // seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.1); //original is 0.05 ->0.3

    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (close_scan_cloud_xyz);
    seg.segment (*inliers, *coefficients);
    //Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (filter_scan_cloud_xyz);
    extract.setIndices (inliers);
    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*close_scan_cloud_xyz);


    // remove ground
    // removeGround(filter_scan_cloud_xyz);
    // Create the filtering object
    // denoise(filter_scan_cloud_xyz);


    // cropbox for map
    submap(filter_cloud_pcl, crop_cloud_pcl, false, 2);

    pcl::fromPCLPointCloud2(*crop_cloud_pcl, *filter_cloud_xyz);
    // denoise(filter_cloud_xyz);
    // pcl::toPCLPointCloud2(*filter_cloud_xyz, *filter_cloud_pcl);
    // pcl_conversions::fromPCL(*crop_cloud_pcl, *filter_cloud_ros);


    filter_cloud_ros->header.frame_id = "map" ;
    map_pub.publish(*filter_cloud_ros);

    pcl::PointCloud<pcl::PointXYZ> final_cloud_xyz;

    // ICP
    icp.setInputSource(filter_scan_cloud_xyz);
    icp.setInputTarget(filter_cloud_xyz);
    if(calibrate_pos){
        icp.setMaxCorrespondenceDistance(10);
        icp.setTransformationEpsilon(1e-6);
        icp.setMaximumIterations(100);
        icp.setEuclideanFitnessEpsilon(1e-3);
    }
    icp.align(final_cloud_xyz, guess);
    if (icp.hasConverged()) {
        prev_score = icp.getFitnessScore();
        ROS_INFO("ICP score: %f", prev_score);
        // cout << "[" << sec << "." << setfill('0') << setw(5) << nsec <<"] ICP score: " << icp.getFitnessScore();
        if(prev_score < 1){
            start_output = true;
            // cout << "START!!!" << endl;
        }
    }


    Eigen::Matrix4f best_trans = icp.getFinalTransformation();
    if(!calibrate_pos) {
        float min_score_ori = icp.getFitnessScore();
        Eigen::Matrix4f trans_ori = icp.getFinalTransformation();
        ROS_INFO("Map-ICP position: (%f,%f,%f)",trans_ori(0,3),trans_ori(1,3),trans_ori(2,3));
  
        t = clock();
        vector<float> score(8,0);
        vector<Eigen::Matrix4f> trans(8,Eigen::Matrix4f::Identity());
        // #pragma omp parallel for
        for(int i=0; i<8; i++) {
            calibratePosition(i, guess, score, trans, icp, 10);
        }
        score.push_back(min_score_ori);
        trans.push_back(trans_ori);


        ROS_INFO("Time cost by for position: %f ms\n",((float)(clock()-t))/CLOCKS_PER_SEC*1000);

        auto min_score_it = min_element(score.begin(), score.end());
        int min_score_ind = min_score_it - score.begin();

        cout << "all score: ";
        for(int i=0; i<score.size(); i++) cout << score[i] << " ";
        cout << endl;

        cout << *min_score_it << "\t" << min_score_ind << endl;
        best_trans = trans[min_score_ind];


        ROS_INFO("Before random find: (%f,%f,%f)",guess(0,3),guess(1,3),guess(2,3));
        ROS_INFO("Best random find: (%f,%f,%f)",best_trans(0,3),best_trans(1,3),best_trans(2,3));
        guess = best_trans;

        calibrate_pos = true;
        return;
    }

    guess = best_trans;//icp.getFinalTransformation();

/*
    //NDT
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setTransformationEpsilon (0.01);
    ndt.setStepSize (0.1);
    ndt.setResolution (1.0);
    ndt.setMaximumIterations (35);
    ndt.setInputSource (filter_scan_cloud_xyz);
    ndt.setInputTarget (filter_cloud_xyz);
    ndt.align (final_cloud_xyz, guess);
    guess = ndt.getFinalTransformation(); 
    cout << " Normal Distributions Transform has converged:" << ndt.hasConverged ()
         << " score: " << ndt.getFitnessScore () << endl;
*/

    // output the scan point cloud after align
    sensor_msgs::PointCloud2 final_cloud_ros;
    pcl::PCLPointCloud2 final_cloud_pcl;
    pcl::toPCLPointCloud2(final_cloud_xyz, final_cloud_pcl);
    pcl_conversions::fromPCL(final_cloud_pcl, final_cloud_ros);
    final_cloud_ros.header.frame_id = "map";
    scan_pub.publish(final_cloud_ros);


    sensor_msgs::PointCloud2 close_scan_cloud_ros;
    pcl::PCLPointCloud2 close_scan_cloud_pcl;
    pcl::toPCLPointCloud2(*close_scan_cloud_xyz, close_scan_cloud_pcl);
    pcl_conversions::fromPCL(close_scan_cloud_pcl, close_scan_cloud_ros);
    close_scan_cloud_ros.header.frame_id = "map";
    scan_filtered_pub.publish(close_scan_cloud_ros);

    //output csv
    if (start_output) {
        sec = scan_msg->header.stamp.sec;
        nsec = scan_msg->header.stamp.nsec;
        float out_x,out_y,out_z;
        out_x = guess(0,3); 
        out_y = guess(1,3); 
        out_z = guess(2,3);
        file.open(result_path,ios::out | ios::app);
        if (nsec >= 100000000)
            file << sec << "." << nsec << ", "<< out_x << ", "<< out_y << ", "<< out_z<< endl;
        else
            file << sec << ".0" << nsec << ", "<< out_x << ", "<< out_y << ", "<< out_z<< endl;
        file.close();
    }

    // publish marker
    geometry_msgs::Point p;
    p.x = guess(0,3);
    p.y = guess(1,3);
    p.z = guess(2,3);
    line_strip.points.push_back(p);
    if(line_strip.points.size()>15000)
        line_strip.points.clear();
    marker_pub.publish(line_strip);

    // send tf
    Eigen::Affine3f tf_f;
    tf_f = guess;
    Eigen::Affine3d tf_d = tf_f.cast<double>();
    static tf::TransformBroadcaster br;
    tf::Transform scan_tf;
    tf::transformEigenToTF(tf_d, scan_tf);
    br.sendTransform(tf::StampedTransform(scan_tf, ros::Time::now(), "map", "scan"));
}

Eigen::Matrix2f transformAngle() {
    Eigen::Matrix2f T;

    float x_angle = 0, y_angle = 0, z_angle = 0;
    for(int i = 1; i < gps_buffer.size() ; i++) {
        float x, y, z;
        x = gps_buffer[i].x - gps_buffer[i-1].x;
        y = gps_buffer[i].y - gps_buffer[i-1].y;
        z = gps_buffer[i].z - gps_buffer[i-1].z;

        z_angle += atan2(x,y);
        x_angle += atan2(y,z);
        y_angle += atan2(z,x);
    }
    x_angle /= gps_buffer.size()-1;
    y_angle /= gps_buffer.size()-1;
    z_angle /= gps_buffer.size()-1;

    x_angle += M_PI/2;
    z_angle += M_PI/6;
    ROS_INFO("x,y,z init angle: %.2f, %.2f, %.2f", x_angle*180/M_PI, y_angle*180/M_PI, z_angle*180/M_PI);

    T << cos (z_angle), sin(z_angle),
        -sin (z_angle), cos (z_angle);

    return T;
}



void pointCallback(const geometry_msgs::Point &msg) {
    gps_buffer.push_back(msg);
    if (gps_buffer.size() >= 4 && gps_buffer.size() < 8) {
        guess.block(0,0,2,2) = transformAngle();
        cout << guess << endl;

        if (!calibrate_pos){
            guess(0,3) = msg.x; 
            guess(1,3) = msg.y;
            guess(2,3) = msg.z;
            ROS_INFO("GPS-Estimated init pos: %f,%f,%f", msg.x, msg.y, msg.z);
        }
        angle_init = true;
    }
}

void initial_marker() {
    line_strip.header.frame_id = "/map";
    line_strip.ns = "linestrip";
    line_strip.action = visualization_msgs::Marker::ADD;
    line_strip.pose.orientation.w = 1.0;
    line_strip.id = 2;
    line_strip.type = visualization_msgs::Marker::LINE_STRIP;


    line_strip.scale.x = 0.5;
    line_strip.color.g = 1.0;
    line_strip.color.a = 1.0;
}

int main(int argc, char** argv) {
    ros::init (argc, argv, "localization_node");
    ros::NodeHandle nh;

    // read parameter and set path
    nh.getParam("file_num", file_num);
    ROS_INFO("file_num: %d", file_num);

    string base_path = ros::package::getPath("localization");

    string data_path = base_path+"/data/map"+std::to_string(file_num)+".pcd";
    cout << "Data dir: " << data_path << endl;
    result_path = base_path+"/output/localization_result"+std::to_string(file_num)+".csv";
    cout << "Result dir: " << result_path << endl;
    file.open(result_path,ios::out);
    if(file) {
        std::remove(result_path.c_str());
    }
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (data_path, *cloud_xyz) == -1) { // load the map file
        PCL_ERROR ("Couldn't read map \n");
        return (-1);
    }

    //convert to ros format
    pcl::toPCLPointCloud2(*cloud_xyz, *cloud_pcl);
    pcl_conversions::fromPCL(*cloud_pcl, *cloud_ros);
    filtering(cloud_pcl, filter_cloud_pcl, 0.25);
    pcl::fromPCLPointCloud2(*filter_cloud_pcl, *filter_cloud_xyz);
    pcl_conversions::fromPCL(*filter_cloud_pcl, *filter_cloud_ros);

    initial_marker();

    //publisher & subscriber
    point_sub = nh.subscribe("/gps_point", 10, pointCallback);
    scan_sub = nh.subscribe("/points_raw", 10, scanCallback);
    map_pub = nh.advertise<sensor_msgs::PointCloud2>("map_pointcloud", 10);
    scan_pub = nh.advertise<sensor_msgs::PointCloud2>("scan_pointcloud", 10);
    scan_filtered_pub = nh.advertise<sensor_msgs::PointCloud2>("scan_filtered_pointcloud", 10);
    marker_pub = nh.advertise<visualization_msgs::Marker>("result_marker",1);

    cout << "map loaded"<<endl;
    ros::spin();
}
