#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/feature2d/feature2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include "/home/kh/ORBSLAM2/include/alglib/optimization.h"

using namespace std;
using namespace eigen;
using namespace cv;
using namespace alglib;


vector<Point3f> BA_map_points;//
vector<Point2f> observations;
vector<Point2d> observations_left;
vector<Point2d> observations_right;
vector<float> optimized_depth;

Matrix3d R_ba;
Vector3d t_ba;

Vector3f rotationMatrixToEulerAngles(Mat& R)
{
  float sy = sqrt(R.at<double>(0, 0)*R.at<double>(0, 0)+R.at<double>(1, 0)*R.at<double>(1, 0));
  bool singular = sy < 1e-6;

  float x, y, z;
  if(!singular)
  {
    x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
	y = atan2(-R.at<double>(2, 0), sy);
	z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
  }
  else
  {
    x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
	y = atan2(-R.at<double>(2, 0), sy);
	z = 0;
  }
  return Vector3f(x,y,z);
}

void find_feature_matches(const Mat& img_1, Mat& img_2, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, vector<Dmatch>& matches);

Point2d pixel2cam(const Point2d& p, const Mat& K);

void bunldeAdjustment(const vector<Point3f> points_3d, const vector<Point2f> points_2d, const Mat& K, Mat& R, Mat& t, Matrix3d& R_ba, Vector3d& t_ba, vector<Point3f>& ba_points);

int iteration_num = 0;

void blei_jac(const real_1d_array& x, double& fi, real_1d_array& jac, void* ptr)
{
  /*int var_length = BA_map_points.size();
  cout << "---------------" << iteration_num << "_________--------" << endl;
  for (int i = 0; i < BA_map_points.size(); i++)
  { 
    cout << x[i] << ", ";
  }
  cout << "---------------" << iteration_num << "_________--------" << endl;
  iteration_num++;*/

  double func = 0.0;
  const float fx = 520.9;
  const float fy = 521.0;
  const float cx = 325.1;
  const float cy = 249.7;

  Eigen::Matrix3f K;
  K << fx,0, cx, 0, fy, cy, 0, 0, 1;

  for(std::size_t i = 0; i < BA_map_points.size(); i++)
  {	

    Point3f map_point_temp(observations_right[j].x/x[j], observations_right[j].y/x[j], 1.0/x[j]);
	cv::Mat xyz_w = (cv::Mat_<double>(3, 1) << map_point_temp.x, map_point_temp.y, map_point_temp.z);
	cv::Mat xyz_trans = xyz_w.clone();

	double px = xyz_trans.at<double>(0);
  	double py = xyz_trans.at<double>(1);
  	double pz = xyz_trans.at<double>(2);
  	double z_2 = pz*pz;

	Matrix<double,3,3> Hp_tmp;
  	Hp_tmp(0,0) = fx;
  	Hp_tmp(0,1) = 0;
  	Hp_tmp(0,2) = -px/pz*fx;
  
  	Hp_tmp(1,0) = 0;
  	Hp_tmp(1,1) = fy;
  	Hp_tmp(1,2) = -py/pz*fy;

	Hp_tmp(2,0) = 0;
	Hp_tmp(2,1) = 0;
	Hp_tmp(2,2) = 1;
	
  	Matrix<double, 3, 3> Hp_ =	-1.0/pz * Hp_tmp;// * Rcw_ei;///此处矩阵后面添加一行
  	//cout << "**************************************Hp_ is  " << Hp_ << endl;
    
  	Matrix<double, 3, 6> Hx_;
  
  	Hx_(0,0) =	px*py/z_2 *fx;
  	Hx_(0,1) = -(1+(px*px/z_2)) *fx;
  	Hx_(0,2) = py/pz *fx;
  	Hx_(0,3) = -1./pz *fx;
  	Hx_(0,4) = 0;
  	Hx_(0,5) = px/z_2 *fx;
  
  	Hx_(1,0) = (1+py*py/z_2) *fy;
  	Hx_(1,1) = -px*py/z_2 *fy;
  	Hx_(1,2) = -px/pz *fy;
  	Hx_(1,3) = 0;
  	Hx_(1,4) = -1./pz *fy;
  	Hx_(1,5) = py/z_2 *fy;

	Hx_(2, 0) = 0;
	Hx_(2, 1) = 0;
	Hx_(2, 2) = 0;
	Hx_(2, 3) = 0;
	Hx_(2, 4) = 0;
	Hx_(2, 5) = 0;

	MatrixXd Hc = MatrixXd::Zero(3, 6);
  	MatrixXd Hp_inv = MatrixXd::Zero(3, 3);
  	MatrixXd H = MatrixXd::Zero(6, 6);

	Hp_inv  = Hp_.inverse();
	Hc = Hp_inv*Hx_;
	H = Hc.transpose()* Hc;

	float tra = H.trace();
	func += tra;
  }
  fi = func;

  for(int g_index = 0; g_index < BA_map_points.size(); g_index++)
  {
    Point3f map_point_temp(observations_right[j].x/x[j], observations_right[j].y/x[j], 1.0/x[j]);
	cv::Mat xyz_w = (cv::Mat_<double>(3, 1) << map_point_temp.x, map_point_temp.y, map_point_temp.z);
	cv::Mat xyz_trans = xyz_w.clone();

	double px = xyz_trans.at<double>(0);
  	double py = xyz_trans.at<double>(1);
  	double pz = xyz_trans.at<double>(2);
  	double z_2 = pz*pz;

	double jac_zc = -z_2((2*pz*(fx + (fx*px*px)/z_2)*(fx + (fx*px*px)/z_2))/(fx*fx) - (2*py*py)/(z_2*pz) - (2*px*px)/(z_2*pz) + (2*pz*(fy + (fy*py*py)/z_2)*(fy + (fy*py*py)/z_2))/(fy*fy) - (4*px*px*py*py)/(z_2*pz) - (4*px*px*(fx + (fx*px*px)/z_2))/(fx*pz) - (4*py*py*(fy + (fy*py*py)/z_2))/(fy*pz));
	jac[g_index] = jac_zc;
  }

}

void LocalMapPointsOptimization()
{
  const float fx = 520.9;
  const float fy = 521.0;
  const float cx = 325.1;
  const float cy = 249.7;
  real_1d_array x;
  real_1d_array s;
  real_2d_array c;
  integer_1d_array ct;
  real_1d_array budl;
  real_1d_array budu;

  budl.setlength(BA_map_points.size());
  budu.setlength(BA_map_points.size());
  x.setlength(BA_map_points.size());
  s.setlength(BA_map_points.size());
  c.setlength(4*BA_map_points.size(),BA_map_points.size() + 1); 
  ct.setlength(4*BA_map_points.size());
  int index = 0;

  for(auto iter:BA_map_points)
  {
    Mat R_ba_cv, t_ba_cv;
	eigen2cv(R_ba, R_ba_cv);
	eigen2cv(t_ba, t_ba_cv);
	Mat map = (Mat_<double>(3, 1) << iter.x, iter.y, iter.z);
	Mat map_c = R_ba_cv*map+ t_ba_cv;
	float px = map_c.at<double>(0);
	float py = map_c.at<double>(1);
	float pz = map_c.at<double>(2);

	  for(int i = 0; i < BA_map_points.size(); i++)
	  {
	    if(i == index)
	    {
	      c[4*index][i] = fx*px;
		  c[4*index+1][i] = fx*px;
		  c[4*index+2][i] = fy*py;
		  c[4*index+3][i] = fy*py;
		}
		else
	    {
	      c[4*index][i] = 0;
		  c[4*index+1][i] = 0;
		  c[4*index+2][i] = 0;
		  c[4*index+3][i] = 0;
		}
	  }

	  c[4*index][BA_map_points.size()] = 3+observations[index].x - cx;
	  c[4*index+1][BA_map_points.size()] = -3+observations[index].x - cx;
	  c[4*index+2][BA_map_points.size()] = 3+observations[index].y - cy;
	  c[4*index+3][BA_map_points.size()] = -3+observations[index].y - cy;
	  
	  ct[4*index] = 1;
	  ct[4*index+1] = 1;
	  ct[4*index+2 = 1;
	  ct[4*index+3] = 1;

	  budl[index] = 0.001;
	  budu[index] = 1000;

	  x[index] = 1.0/pz;
	  s[index] = 1.0/pz;
	  index += 1;
  }

  minbleicstate state;
  double epsg = 0.0;
  double epsf = 0;
  double epsx = 0.0000001;
  ae_int_t maxits = 0;
  minbleiccreate(x, state);
  minbleicsetlc(state, c, ct);
  minbleicsetbc(state, budl, budu);
  minbleicsetscale(state, s);
  minbleicsetcond(state, epsg, epsf, epsx, maxits);

  real_1d_array x1;
  minbleicreport rep;

  alglib::minbleicoptimize(state, bleic_jac);
  minbleicresults(state, x1, rep);

  for(int i = 0; i < BA_map_points.size(); i++)
  {
     optimized_depth.push_back(x1[i]);
  }
}

int main(int argc, char** argv)
{
  Mat img_1 = imread("../1.png", CV_LOAD_IMAGE_COLOR);
  Mat img_2 = imread("../3.png", CV_LOAD_IMAGE_COLOR);

  Eigen::Matrix3d R1;
  Eigen::Quaterniond quaternion1(-0.3579, 0.6331, -0.5995, 0.3341);
  R1 = quaternion1.matrix();

  Eigen::Matrix3d R2;
  Eigen::Quaterniond quaternion2(-0.4083, 0.6767, -0.5551, 0.2593);
  R2 = quaternion2.matrix();

  Vector3d t1(0.0388, -1.0918, 1.4542);
  Vector3d t2(0.0511, -1.0731, 1.4442);
  Matrix3d gt_R = R2.transpose()*R1;
  Matrix3d gt_t = R2.transpose()*t1 - R2.transpose()*t2;
  cout << "Ground truth is (x-y-z)" << gt_R.eulerAngles(0, 1, 2)/3.14*180 << endl;
  cout << "Translation Ground truth is" << gt_t << endl;

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);

  Mat d1 = imread("../1_depth.png", CV_LOAD_IMAGE_UNCHANGED);
  Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 0, 249.7, 0, 0, 1);
  vector<Point3f> pts_3d;
  vector<Point2f> pts_2d;
  vector<float> depth_gt;
  int random_num = 0;

  for(DMatch m:matches)
  {
    ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
	if(d ==0)
		continue;
	float dd = 1.0 *d / 5000.0;
	Point2f p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);

	pts_3d.push_back(Point3f(1.0 * p1.x*dd + 0.0, 1.0 * p1.y*dd + 0.0, 1.0*dd));
	pts_2d.push_back(keypoints_2[m.trainIdx].pt);
	observations_left.push_back(p1);
	observations_right.push_back(pixel2cam(keypoints_2[m.trainIdx].pt, K));
	observations.push_back(keypoints_2[m.trainIdx].pt);
	depth_gt.push_back(d/5000.0);
	random_num++;
  }

  Mat r, t;
  solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);
  Mat R;
  cv::Rodrigues(r, R);
  Vector3f R_xyz_pnp;
  R_xyz_pnp = rotationMatrixToEulerAngles(R);
  cout << "Results of pnp" << endl;
  cout << "angles of x y z are" << R_xyz_pnp/3.14*180 << endl;
  cout << "translation is " << t << endl;

  chrono::steady_clock::time_point t1_ba = chrono::steady_clock::now();
  bundleAdjustment(pts_3d, pts_2d, K, R, t, R_ba, t_ba, BA_map_points);
  chrono::steady_clock::time_point t2_ba = chrono::steady_clock::now();
  chrono::duration<double> time_used_ba = chrono::duration_cast<chrono::duration<double>>(t2_ba-t1_ba);

  Vector3d R_xyz_ba;
  R_xyz_ba = R_ba.eulerAngles(0, 1, 2);
  cout << "results of BA" << endl;
  cout << "angles of x y z are" << R_xyz_ba/3.14*180 << endl;
  cout << "translation is " << t_ba << endl;
  chrono::steady_clock::time_point t1_c = chrono::steady_clock::now();
  LocalMapPointsOptimization();
  chrono::steady_clock::time_point t2_c = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2_c-t1_c);

  for(int i = 0; i < BA_map_points; i++)
  {
    double depth_g = gt_R(2, 0)*observations_left[i].x*depth_gt[i] + gt_R(2, 1)* observations_left[i].y*depth_gt[i] + gt_R(2, 2)*depth_gt[i]+gt_t(2);

	Vector3d map_point_w(BA_map_points[i].x, BA_map_points[i].y, BA_map_points[i].z);
	Vector3d pc_ba = gt_R*map_point_w + gt_t;
  }
  
  
}

void find_feature_matches(const Mat & img_1, Mat & img_2, vector < KeyPoint > & keypoints_1, vector < KeyPoint > & keypoints_2, vector < Dmatch > & matches)
{
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();

  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

  detector->detec(img_1, keypoints_1);
  detector->detec(img_2, keypoints_2);

  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  vector<DMatch> match;

  matcher->match(descriptors_1, descriptors_2, match);

  double min_dist = 10000, max_dist = 0;

  for(int i = 0; i < descriptors_1.rows; i++)
  {
    double dist = match[i].distance;
	if(dist < min_dist)
		min_dist = dist;
	if(dist > max_dist)
		max_dist = dist;
  }

  for(int i = 0; i < descriptors_1.rows;; i++)
  {
    if(match[i].distance <= max(2*min_dist, 30.0))
		matches.push_back(match[i]);
  }
}

cv::Point2d pixel2cam(const cv::Point2f& p,cv::Mat& K){
    return cv::Point2d( (p.x - K.at<double>(0,2)) / K.at<double>(0,0),
                        (p.y - K.at<double>(1,2)) / K.at<double>(1,1)
            );
}



void bundleAdjustment(const vector<cv::Point3f> points_3d,
                      const vector<cv::Point2f> points_2d,
                      const cv::Mat& K,
                      cv::Mat& R,
                      cv::Mat& t, Matrix3d& R_ba, Vector3d& t_ba,vector<Point3f>& ba_points)
{
    if(!ba_points.empty())
		ba_points.clear();
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;
    //CSPARSE是一个求解线性问题的矩阵库
    Block::LinearSolverType* lineasolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(unique_ptr<Block::LinearSolverType>(lineasolver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(unique_ptr<Block>(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    //顶点
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();//相机位姿
    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
             R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
             R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2);

    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat,Eigen::Vector3d(t.at<double>(0,0),t.at<double>(0,1),t.at<double>(0,2))));
    optimizer.addVertex(pose);

    int index = 1;
    for (const cv::Point3f p : points_3d){
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    //设定相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double>(0,0),Eigen::Vector2d(K.at<double>(0,2),K.at<double>(1,2)),0);

    camera->setId(0);
    optimizer.addParameter(camera);


    //边
    index = 1;
    for(const cv::Point2f p : points_2d){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        //将边与点一一对应？
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
        edge->setVertex(1,pose);
        edge->setMeasurement(Eigen::Vector2d(p.x,p.y));
        //以上面相机参数为准
        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
    }
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    //最多100次迭代
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "g2o图优化耗时：" << time_used.count() << "秒." << endl;

    cout << "经过g2o图优化之后的欧氏变换矩阵T：" << endl;
    cout << Eigen::Isometry3d(pose->estimate()).matrix() << endl;

	g2o::SE3Quat T_se = pose->estimate();
	Eigen::Matrix<double, 4, 4> T = T_se.to_homogeneous_matrix();
	R_ba = T.block<3,3>(0, 0);
	Matrix<double, 3, 1> t_ba_ = T.block<3, 1>(0, 3);
	t_ba(0) = t_ba_(0);
	t_ba(1) = t_ba_(1);
	t_ba(2) = t_ba_(2);

	for(size_t i = 0; i < points_3d,size(); i++)
	{
	  g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ* >(optimizer.vertex(i+1));
	  Eigen::Vector3d pose = v->estimate();
	  Point3f temp_point(pose(0), pose(1), pose(2));
	  ba_points.push_back(temp_point);
	}
}













