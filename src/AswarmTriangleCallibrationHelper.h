// MIT License

// Copyright (c) 2024 Muhammad Khalis bin Mohd Fadil

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/registration/icp.h>  // For ICP matching
#include <pcl/registration/gicp.h>  // For ICP matching
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/common/pca.h>

#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/common/common.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/common/angles.h>
#include <pcl/filters/passthrough.h>

#include <omp.h>
#include <cmath>  
#include <limits>
#include <iostream>
#include <random>
#include <deque>
#include <vector>

#include <Eigen/Dense>

// HELPER FUNCTION

// Function to convert raw float array to PCL Point Cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr convertToPointCloud(const float* inputCloud, uint32_t numInputCloud, uint32_t constantSize) {
    // Create a point cloud to store the converted points
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->width = numInputCloud;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(numInputCloud);

    // Parallel loop to convert the float array to a PCL point cloud
    #pragma omp parallel for
    for (uint32_t i = 0; i < numInputCloud; ++i) {
        cloud->points[i].x = inputCloud[i];                       // X values
        cloud->points[i].y = inputCloud[i + constantSize];        // Y values
        cloud->points[i].z = inputCloud[i + constantSize * 2];    // Z values
    }

    return cloud;
}

// Function to downsample the point cloud using VoxelGrid filter
pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, 
                                                        float leafSize) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(inputCloud);
    vg.setLeafSize(leafSize, leafSize, leafSize);
    vg.filter(*filteredCloud);
    return filteredCloud;
}

// Function to find and return the nearest cluster using Euclidean Cluster Extraction
pcl::PointCloud<pcl::PointXYZ>::Ptr findNearestCluster(pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud, 
                                                       float cluster_tolerance, 
                                                       uint32_t min_cluster_size, 
                                                       uint32_t max_cluster_size) {
    // Step 1: Set up KdTree for clustering
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(filteredCloud);

    // Step 2: Perform Euclidean Cluster Extraction
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance);  // Distance threshold for clustering
    ec.setMinClusterSize(min_cluster_size);     
    ec.setMaxClusterSize(max_cluster_size);     
    ec.setSearchMethod(tree);
    ec.setInputCloud(filteredCloud);
    ec.extract(cluster_indices);

    // Step 3: Find the nearest cluster based on centroid distance to origin
    pcl::PointCloud<pcl::PointXYZ>::Ptr nearestCluster(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointIndices::Ptr nearest_cluster_indices_ptr(new pcl::PointIndices);

    float min_distance = std::numeric_limits<float>::max();

    #pragma omp parallel
    {
        float thread_min_distance = std::numeric_limits<float>::max();
        pcl::PointIndices::Ptr thread_nearest_cluster_indices_ptr(new pcl::PointIndices);

        // Parallelize centroid calculation and nearest cluster detection
        #pragma omp for nowait
        for (uint32_t i = 0; i < cluster_indices.size(); ++i) {
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*filteredCloud, cluster_indices[i].indices, centroid);
            float distance_to_origin = centroid.norm();

            if (distance_to_origin < thread_min_distance) {
                thread_min_distance = distance_to_origin;
                *thread_nearest_cluster_indices_ptr = cluster_indices[i];
            }
        }

        // Update the nearest cluster in a critical section
        #pragma omp critical
        {
            if (thread_min_distance < min_distance) {
                min_distance = thread_min_distance;
                *nearest_cluster_indices_ptr = *thread_nearest_cluster_indices_ptr;
            }
        }
    }

    // Step 4: Extract the nearest cluster points
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(filteredCloud);
    extract.setIndices(nearest_cluster_indices_ptr);
    extract.setNegative(false);  // Extract the points inside the cluster
    extract.filter(*nearestCluster);

    // Return the nearest cluster cloud
    return nearestCluster;
}

// Function to extract planar and corner features based on normal estimation
std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> 
extractFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr vehicleCloud, float radiusSearch, 
                float curvatureThresholdPlane, float curvatureThresholdCorner) {
    
    // Step 1: Compute normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree2);
    ne.setInputCloud(vehicleCloud);
    ne.setRadiusSearch(radiusSearch);  // Set the radius for normal estimation
    ne.compute(*cloud_normals);

    // Containers for planar and corner features
    pcl::PointCloud<pcl::PointXYZ>::Ptr PlaneFeatures(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr CornerFeatures(new pcl::PointCloud<pcl::PointXYZ>);

    // Thread-local storage for parallel processing
    #pragma omp parallel
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr localPlaneFeatures(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr localCornerFeatures(new pcl::PointCloud<pcl::PointXYZ>);

        #pragma omp for
        for (uint32_t i = 0; i < vehicleCloud->points.size(); ++i) {
            // Feature extraction based on curvature
            if (cloud_normals->points[i].curvature < curvatureThresholdPlane) {
                // Planar feature
                localPlaneFeatures->points.emplace_back(vehicleCloud->points[i]);
            } else if (cloud_normals->points[i].curvature > curvatureThresholdCorner) {
                // Corner feature
                localCornerFeatures->points.emplace_back(vehicleCloud->points[i]);
            }
        }

        // Merge local results into global result (critical section for merging)
        #pragma omp critical
        {
            *PlaneFeatures += *localPlaneFeatures;
            *CornerFeatures += *localCornerFeatures;
        }
    }

    // Return both the plane and corner features as a pair
    return std::make_pair(PlaneFeatures, CornerFeatures);
}

// Function to find the plane coefficients using RANSAC
Eigen::Vector3f findPlaneCoefficients(pcl::PointCloud<pcl::PointXYZ>::Ptr planeFeatures, float distanceThreshold = 0.01f) {
    // Step 1: Use RANSAC to segment the plane
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distanceThreshold);
    seg.setInputCloud(planeFeatures);
    seg.segment(*inliers, *coefficients);

    // Check if plane segmentation succeeded
    if (inliers->indices.empty()) {
        // std::cerr << "Could not estimate a plane model for the given point cloud." << std::endl;
        return Eigen::Vector3f::Zero();  // Return zero vector if segmentation fails
    }

    // Step 2: Extract the normal vector coefficients (a, b, c)
    float a = coefficients->values[0];      //direction of the plane’s normal vector
    float b = coefficients->values[1];
    float c = coefficients->values[2];

    // Return the plane normal as an Eigen::Vector3f
    Eigen::Vector3f normal(a, b, c);
    return normal.normalized();  // Return normalized normal vector
}

// Function to find the corners of a known-dimension rectangular plane using PCA
std::vector<Eigen::Vector3f> findRectangleCornersWithPCA(pcl::PointCloud<pcl::PointXYZ>::Ptr planeCloud, float longerSide, float shorterSide) {
    std::vector<Eigen::Vector3f> rectangleCorners;

    // Step 1: Perform PCA to find the principal axes and mean (center) of the point cloud
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(planeCloud);

    // Get the mean and eigenvectors (principal directions)
    Eigen::Vector3f center = pca.getMean().head<3>();
    Eigen::Matrix3f eigenVectors = pca.getEigenVectors();

    // Define the primary axes of the plane based on PCA
    Eigen::Vector3f axis1 = eigenVectors.col(0); // First principal direction (e.g., width)
    Eigen::Vector3f axis2 = eigenVectors.col(1); // Second principal direction (e.g., height)

    // Step 2: Calculate the corner points using the center, axis1, and axis2
    Eigen::Vector3f corner1 = center + (longerSide / 2.0 * axis1) + (shorterSide / 2.0 * axis2);  
    Eigen::Vector3f corner2 = center - (longerSide / 2.0 * axis1) + (shorterSide / 2.0 * axis2);  
    Eigen::Vector3f corner3 = center - (longerSide / 2.0 * axis1) - (shorterSide / 2.0 * axis2);  
    Eigen::Vector3f corner4 = center + (longerSide / 2.0 * axis1) - (shorterSide / 2.0 * axis2);  

    // Step 3: Store the corners in the vector
    rectangleCorners.push_back(corner1);
    rectangleCorners.push_back(corner2);
    rectangleCorners.push_back(corner3);
    rectangleCorners.push_back(corner4);

    return rectangleCorners;
}

// Function to interpolate points along an edge
pcl::PointCloud<pcl::PointXYZ>::Ptr interpolatePoints(const Eigen::Vector3f& start, const Eigen::Vector3f& end, float point_spacing) {
    // Create a point cloud to hold the interpolated points
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Calculate the distance between the start and end points
    float distance = (end - start).norm();
    
    // Calculate the number of points to interpolate based on the point spacing
    int num_points = std::ceil(distance / point_spacing);

    // Interpolate points along the edge
    for (int i = 0; i <= num_points; ++i) {
        float t = static_cast<float>(i) / num_points;
        Eigen::Vector3f point = start + t * (end - start);

        // Add the interpolated point to the point cloud
        pointCloud->points.emplace_back(point[0], point[1], point[2]);
    }

    // Set the cloud properties
    pointCloud->width = pointCloud->points.size();
    pointCloud->height = 1;
    pointCloud->is_dense = true;

    return pointCloud;
}

// Function to convert Eigen::Vector3f to PCL PointXYZ
pcl::PointXYZ eigenToPointXYZ(const Eigen::Vector3f& point) {
    return pcl::PointXYZ(point[0], point[1], point[2]);
}

// Function to convert PCL PointXYZ to Eigen::Vector3f
Eigen::Vector3f pointXYZToEigen(const pcl::PointXYZ& point) {
    return Eigen::Vector3f(point.x, point.y, point.z);
}

// Function to convert Eigen::Vector4f to Eigen::Vector3f
Eigen::Vector3f convertToVector3f(const Eigen::Vector4f& vec4) {
    return Eigen::Vector3f(vec4[0], vec4[1], vec4[2]);
}

// Function to calculate the distance between two points
float calculateDistance(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) {
    return (p1 - p2).norm();
}

float calculateAngle(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
    // Calculate the angle using normalized vectors
    float cosTheta = v1.normalized().dot(v2.normalized());
    cosTheta = std::clamp(cosTheta, -1.0f, 1.0f); // Clamp to handle any minor floating-point inaccuracies
    return std::acos(cosTheta) * 180.0f / M_PI; // Convert radians to degrees
}

// Function to calculate the score of a triangle with tolerance
float calculateTriangleScore(const Eigen::Vector3f& pointA, const Eigen::Vector3f& pointB, const Eigen::Vector3f& pointC,
                             float distTolerance, float angleTolerance, 
                             float idealAdjacentSide, float idealOppositeSide, float idealHypotenuse,
                             float idealAngleASOS, float idealAngleOSH, float idealAngleHAS) {

    // Calculate distances
    float dAB = calculateDistance(pointA, pointB);
    float dBC = calculateDistance(pointB, pointC);
    float dCA = calculateDistance(pointC, pointA);

    float dAdjacentSide;
    float dOppositeSide;
    float dHypotenuse;
    float angleASOS;
    float angleOSH;
    float angleHAS;

    if (dAB >= dBC){
        dAdjacentSide = dAB;
        dOppositeSide = dBC;
        dHypotenuse = dCA;
        // Calculate angles
        angleASOS = calculateAngle(pointB - pointA, pointC - pointB); 
        angleOSH = calculateAngle(pointC - pointB, pointA - pointC);  
        angleHAS = calculateAngle(pointA - pointC, pointB - pointA);  
    }else{
        dAdjacentSide = dBC;
        dOppositeSide = dAB;
        dHypotenuse = dCA;
        // Calculate angles
        angleASOS = calculateAngle(pointB - pointA, pointC - pointB);  
        angleOSH = calculateAngle(pointA - pointC, pointB - pointA);  
        angleHAS = calculateAngle(pointC - pointB, pointA - pointC);  
    }
    
    // Calculate score based on how close the distances are to the ideal values
    float distanceScore = 0.0;
    distanceScore += std::max(0.0f, std::abs(dAdjacentSide - idealAdjacentSide) - distTolerance);
    distanceScore += std::max(0.0f, std::abs(dOppositeSide - idealOppositeSide) - distTolerance);
    distanceScore += std::max(0.0f, std::abs(dHypotenuse - idealHypotenuse) - distTolerance);

    // Calculate score based on how close the angles are to the ideal values
    float angleScore = 0.0;
    angleScore += std::max(0.0f, std::abs(angleASOS - idealAngleASOS) - angleTolerance);
    angleScore += std::max(0.0f, std::abs(angleOSH - idealAngleOSH) - angleTolerance);
    angleScore += std::max(0.0f, std::abs(angleHAS - idealAngleHAS) - angleTolerance);

    // The final score is the sum of the penalties for distance and angle deviations
    float finalScore = distanceScore + angleScore;

    return finalScore;
}

// Function to find candidate points using KdTree
std::vector<Eigen::Vector3f> findCandidatePoints(const Eigen::Vector3f& origin, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                                 pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, float searchRadius) {
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    std::vector<Eigen::Vector3f> candidatePoints;

    pcl::PointXYZ searchPoint = eigenToPointXYZ(origin);
    if (kdtree.radiusSearch(searchPoint, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
        for (int idx : pointIdxRadiusSearch) {
            candidatePoints.push_back(pointXYZToEigen((*cloud)[idx]));
        }
    }

    return candidatePoints;
}

std::tuple<std::vector<Eigen::Vector3f>, float, int> findBestTriangle(const Eigen::Vector3f& originA, const Eigen::Vector3f& originB, 
                                                                        const Eigen::Vector3f& originC, const Eigen::Vector3f& originD,
                                                                        pcl::PointCloud<pcl::PointXYZ>::Ptr candidatePointsCloud, float searchRadius,
                                                                        float distTolerance, float angleTolerance,
                                                                        float adjacentSide, float oppositeSide, float hypotenuse,
                                                                        float adjacentAngle, float vertexAngle, float oppositeAngle) {
    float bestScore = std::numeric_limits<float>::max();
    Eigen::Vector3f best0, best1, best2;
    int bestIndexSource = -1;  

    // Step 1: Build the KdTree
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(candidatePointsCloud);

    // Step 2: Find candidate points
    std::vector<Eigen::Vector3f> candidatesA = findCandidatePoints(originA, candidatePointsCloud, kdtree, searchRadius);
    std::vector<Eigen::Vector3f> candidatesB = findCandidatePoints(originB, candidatePointsCloud, kdtree, searchRadius);
    std::vector<Eigen::Vector3f> candidatesC = findCandidatePoints(originC, candidatePointsCloud, kdtree, searchRadius);
    std::vector<Eigen::Vector3f> candidatesD = findCandidatePoints(originD, candidatePointsCloud, kdtree, searchRadius);

    #pragma omp parallel
    {
        float localBestScore = std::numeric_limits<float>::max();
        Eigen::Vector3f localBest0, localBest1, localBest2;
        int localBestIndexSource = -1;

        // CandidateB as middle point
        #pragma omp for
        for (uint32_t i = 0; i < candidatesB.size(); ++i) {
            const auto& candidateB = candidatesB[i];
            for (const auto& candidateA : candidatesA) {
                for (const auto& candidateC : candidatesC) {
                    float score = calculateTriangleScore(candidateA, candidateB, candidateC, 
                                                         distTolerance, angleTolerance,
                                                         adjacentSide, oppositeSide, hypotenuse, 
                                                         adjacentAngle, vertexAngle, oppositeAngle);
                    if (score < localBestScore) {
                        localBestScore = score;
                        localBest0 = candidateA;
                        localBest1 = candidateB;
                        localBest2 = candidateC;
                        localBestIndexSource = 1;
                    }
                }
            }
        }

        // CandidateC as middle point
        #pragma omp for
        for (uint32_t i = 0; i < candidatesC.size(); ++i) {
            const auto& candidateC = candidatesC[i];
            for (const auto& candidateB : candidatesB) {
                for (const auto& candidateD : candidatesD) {
                    float score = calculateTriangleScore(candidateB, candidateC, candidateD, 
                                                         distTolerance, angleTolerance,
                                                         adjacentSide, oppositeSide, hypotenuse,
                                                         adjacentAngle, vertexAngle, oppositeAngle);
                    if (score < localBestScore) {
                        localBestScore = score;
                        localBest0 = candidateB;
                        localBest1 = candidateC;
                        localBest2 = candidateD;
                        localBestIndexSource = 2;
                    }
                }
            }
        }

        // CandidateD as middle point
        #pragma omp for
        for (uint32_t i = 0; i < candidatesD.size(); ++i) {
            const auto& candidateD = candidatesD[i];
            for (const auto& candidateC : candidatesC) {
                for (const auto& candidateA : candidatesA) {
                    float score = calculateTriangleScore(candidateC, candidateD, candidateA, 
                                                         distTolerance, angleTolerance,
                                                         adjacentSide, oppositeSide, hypotenuse, 
                                                         adjacentAngle, vertexAngle, oppositeAngle);
                    if (score < localBestScore) {
                        localBestScore = score;
                        localBest0 = candidateC;
                        localBest1 = candidateD;
                        localBest2 = candidateA;
                        localBestIndexSource = 3;
                    }
                }
            }
        }

        // CandidateA as middle point
        #pragma omp for
        for (uint32_t i = 0; i < candidatesA.size(); ++i) {
            const auto& candidateA = candidatesA[i];
            for (const auto& candidateD : candidatesD) {
                for (const auto& candidateB : candidatesB) {
                    float score = calculateTriangleScore(candidateD, candidateA, candidateB, 
                                                         distTolerance, angleTolerance,
                                                         adjacentSide, oppositeSide, hypotenuse,
                                                         adjacentAngle, vertexAngle, oppositeAngle);
                    if (score < localBestScore) {
                        localBestScore = score;
                        localBest0 = candidateD;
                        localBest1 = candidateA;
                        localBest2 = candidateB;
                        localBestIndexSource = 4;
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (localBestScore < bestScore) {
                bestScore = localBestScore;
                best0 = localBest0;
                best1 = localBest1;
                best2 = localBest2;
                bestIndexSource = localBestIndexSource;
            }
        }
    }

    std::vector<Eigen::Vector3f> best012 = { best0, best1, best2 };
    return std::make_tuple(best012, bestScore, bestIndexSource);
}


// Function to recorrect triangle based on ideal distances and angles
std::vector<Eigen::Vector3f> recorrectTriangle(const std::vector<Eigen::Vector3f>& bestPoints, int sourceIndex,
                                                float idealAdjacentSide, float idealOppositeSide, float idealHypotenuse,
                                                float idealAngleASOS, float idealAngleOSH, float idealAngleHAS) {
    const float tolerance = 0.0001f;  // Tolerance for distance and angle correction
    Eigen::Vector3f A,B,C;

    // Initialize corrected points with the current best points
    if (sourceIndex == 1 || sourceIndex == 3){
        A = bestPoints[0];
        B = bestPoints[1];
        C = bestPoints[2];
    }else if (sourceIndex == 2 || sourceIndex == 4) {
        A = bestPoints[2];
        B = bestPoints[1];
        C = bestPoints[0];
    }

    // Iteratively adjust until within tolerance
    int maxIterations = 5;
    for (int iter = 0; iter < maxIterations; ++iter) {
        // Calculate current distances
        float currentAB = calculateDistance(A, B);
        float currentBC = calculateDistance(B, C);
        float currentCA = calculateDistance(C, A);

        // Calculate current angles
        float currentAngleABBC = calculateAngle(B - A, C - B);
        float currentAngleBCCA = calculateAngle(C - B, A - C);
        float currentAngleCAAB = calculateAngle(A - C, B - A);

        // Check if all distances and angles are within tolerance
        if (std::abs(currentAB - idealAdjacentSide) < tolerance &&
            std::abs(currentBC - idealOppositeSide) < tolerance &&
            std::abs(currentCA - idealHypotenuse) < tolerance &&
            std::abs(currentAngleABBC - idealAngleASOS) < tolerance &&
            std::abs(currentAngleBCCA - idealAngleOSH) < tolerance &&
            std::abs(currentAngleCAAB - idealAngleHAS) < tolerance) {
            break;  // Exit loop if all within tolerance
        }
        
        // Adjust points to match ideal distances
        if(std::abs(currentCA - idealHypotenuse) > tolerance){
            Eigen::Vector3f directionAC = (C - A).normalized();
            C = A + directionAC * idealHypotenuse;  // Adjust C along the line towards the ideal AC length
        }
        if(std::abs(currentBC - idealOppositeSide) > tolerance){
            Eigen::Vector3f directionCB = (B - C).normalized();
            B = C + directionCB * idealOppositeSide;  // Adjust B along the line towards the ideal CB length
        }
        if(std::abs(currentAB - idealAdjacentSide) > tolerance){
            Eigen::Vector3f directionBA = (A - B).normalized();
            A = B + directionBA * idealAdjacentSide;  // Adjust A along the line towards the ideal BA length
        }

    }

    // Return the corrected points
    return { A, B, C };
}


// ############################################
// Function to apply translation to align the rectangle center with the bounding box center
std::vector<Eigen::Vector3f> applyTranslationToCorners(const std::vector<Eigen::Vector3f>& corners,
                                                       const Eigen::Vector3f& targetCenter) {
    // Calculate the current center of the rectangle
    Eigen::Vector3f rectangleCenter = (corners[0] + corners[1] + corners[2] + corners[3]) / 4.0f;

    // Calculate the translation offset
    Eigen::Vector3f totalOffset = (targetCenter - rectangleCenter) * 0.5f;  // Adaptive scaling

    // Apply translation to each corner
    std::vector<Eigen::Vector3f> translatedCorners;
    for (const auto& corner : corners) {
        translatedCorners.push_back(corner + totalOffset);
    }

    return translatedCorners;
}

// Function to rotate rectangle corners in the local XY plane of a tilted input plane
std::vector<Eigen::Vector3f> applyRotationToCorners(const std::vector<Eigen::Vector3f>& corners, 
                                                    const Eigen::Vector3f& center, 
                                                    const Eigen::Vector3f& planeNormal, 
                                                    float angleDegrees) {
    // Convert angle to radians
    float radians = angleDegrees * M_PI / 180.0f;

    // Step 1: Calculate the rotation needed to align the plane's normal to the global Z-axis
    Eigen::Vector3f globalZ(0, 0, 1);
    Eigen::Quaternionf alignToZ;
    alignToZ.setFromTwoVectors(planeNormal, globalZ);
    Eigen::Matrix3f alignToZMatrix = alignToZ.toRotationMatrix();

    // Step 2: Rotate in the aligned (local) XY plane around the Z-axis
    Eigen::Matrix3f yawRotation;
    yawRotation << cos(radians), -sin(radians), 0,
                   sin(radians),  cos(radians), 0,
                   0,             0,            1;

    // Step 3: Combine transformations: first align to Z, then rotate, then align back to original plane orientation
    Eigen::Matrix3f alignBack = alignToZMatrix.transpose();  // inverse rotation
    Eigen::Matrix3f totalTransformation = alignBack * yawRotation * alignToZMatrix;

    // Step 4: Apply the total transformation to each corner point
    std::vector<Eigen::Vector3f> rotatedCorners;
    for (const auto& corner : corners) {
        rotatedCorners.push_back(totalTransformation * (corner - center) + center);
    }

    return rotatedCorners;
}

// Function to calculate shortest distance from a point to a line segment
float pointToLineDistance(const Eigen::Vector2f& point, const Eigen::Vector2f& lineStart, const Eigen::Vector2f& lineEnd) {
    Eigen::Vector2f lineDir = lineEnd - lineStart;
    float lineLengthSquared = lineDir.squaredNorm();
    
    if (lineLengthSquared == 0) return (point - lineStart).norm();  // Line is a single point

    float t = std::max(0.0f, std::min(1.0f, (point - lineStart).dot(lineDir) / lineLengthSquared));
    Eigen::Vector2f projection = lineStart + t * lineDir;
    return (point - projection).norm();
}

bool isPointInsideRectangle(const Eigen::Vector2f& point, const std::vector<Eigen::Vector2f>& rectangleCorners) {

    // Calculate the cross product for the first edge to set the reference sign
    Eigen::Vector2f firstEdgeDir = rectangleCorners[1] - rectangleCorners[0];
    Eigen::Vector2f firstPtDir = point - rectangleCorners[0];
    float firstCrossProduct = firstEdgeDir.x() * firstPtDir.y() - firstEdgeDir.y() * firstPtDir.x();

    // Determine the sign based on the first cross product (positive or negative)
    bool isPositive = (firstCrossProduct > 0);

    // Check all other edges
    for (uint32_t i = 1; i < rectangleCorners.size(); ++i) {
        Eigen::Vector2f edgeStart = rectangleCorners[i];
        Eigen::Vector2f edgeEnd = rectangleCorners[(i + 1) % rectangleCorners.size()];
        Eigen::Vector2f edgeDir = edgeEnd - edgeStart;
        Eigen::Vector2f ptDir = point - edgeStart;

        float crossProduct = edgeDir.x() * ptDir.y() - edgeDir.y() * ptDir.x();

        // If the cross product sign doesn't match the first one, the point is outside
        if ((crossProduct > 0) != isPositive) {
            return false;
        }
    }
    return true;
}

float calculateBoundaryScore(const pcl::PointCloud<pcl::PointXYZ>::Ptr& planeCloud,
                             const std::vector<Eigen::Vector3f>& rectangleCorners,
                             const Eigen::Vector3f& planeNormal) {
    float score = 0.0f;

    // Step 1: Calculate the rotation matrix to align the plane normal with the global Z-axis
    Eigen::Vector3f globalZ(0.0f, 0.0f, 1.0f);
    Eigen::Quaternionf rotation;
    rotation.setFromTwoVectors(planeNormal, globalZ);
    Eigen::Matrix3f rotationMatrix = rotation.toRotationMatrix();

    // Step 2: Transform rectangle corners to the local coordinate system
    std::vector<Eigen::Vector2f> rectangleCornersLocalXY;
    for (const auto& corner : rectangleCorners) {
        Eigen::Vector3f rotatedCorner = rotationMatrix * corner;
        rectangleCornersLocalXY.emplace_back(rotatedCorner.x(), rotatedCorner.y());
    }

    // Step 3: Iterate through each point in planeCloud and transform it to the local coordinate system
    #pragma omp parallel for reduction(+:score)  // Parallelize with reduction to sum scores across threads
    for (int i = 0; i < planeCloud->points.size(); ++i) {
        const auto& point = planeCloud->points[i];
        Eigen::Vector3f pointVec(point.x, point.y, point.z);
        Eigen::Vector3f rotatedPoint = rotationMatrix * pointVec;
        Eigen::Vector2f ptLocalXY(rotatedPoint.x(), rotatedPoint.y());

        // Step 4: Check if point is inside the rectangle in the local XY plane
        if (isPointInsideRectangle(ptLocalXY, rectangleCornersLocalXY)) {
            continue;  // Skip points inside the boundary, as they don’t contribute to the score
        }

        // Step 5: Calculate minimum distance to any edge for points outside
        float minDistance = std::numeric_limits<float>::max();
        for (uint32_t j = 0; j < rectangleCornersLocalXY.size(); ++j) {
            const Eigen::Vector2f& edgeStart = rectangleCornersLocalXY[j];
            const Eigen::Vector2f& edgeEnd = rectangleCornersLocalXY[(j + 1) % rectangleCornersLocalXY.size()];
            float distance = pointToLineDistance(ptLocalXY, edgeStart, edgeEnd);
            minDistance = std::min(minDistance, distance);  // Track minimum distance to any edge
        }

        // Step 6: Accumulate minimum distance for points outside the boundary
        score += minDistance;
    }
    return score;  // Higher score indicates more points outside or farther from boundary
}

std::vector<Eigen::Vector3f> readjustCornerPoints(const std::vector<Eigen::Vector3f>& cornerPoints,
                                                  pcl::PointCloud<pcl::PointXYZ>::Ptr planeCloud,
                                                  const Eigen::Vector3f& planeNormal, uint32_t type,
                                                  float& boundaryScore, float angleStep, float angleTotal, int maxIterations) {
    constexpr uint32_t triangle = 1;
    constexpr uint32_t rectangle = 2;
    std::vector<Eigen::Vector3f> adjustedCornerPoints;

    if (type == triangle) {
        Eigen::Vector3f A = cornerPoints[0];
        Eigen::Vector3f B = cornerPoints[1];
        Eigen::Vector3f C = cornerPoints[2];

        // Calculate vectors AB and AC
        Eigen::Vector3f AB = B - A;
        Eigen::Vector3f AC = C - A;

        // Calculate the vector BC for distance reference
        Eigen::Vector3f BC = C - B;
        float targetLength = BC.norm();  // Target length for AD to match BC

        // Calculate the normal vector to the plane defined by AB and AC
        Eigen::Vector3f normal = AB.cross(AC).normalized();

        // Use the cross product to find a direction perpendicular to AB in the plane of ABC
        Eigen::Vector3f AD_dir = normal.cross(AB).normalized();

        // Scale AD_dir to have the same length as BC
        Eigen::Vector3f AD = AD_dir * targetLength;

        // Calculate point D by starting at A and adding the AD vector
        Eigen::Vector3f D = A + AD;

        adjustedCornerPoints = {A, B, C, D};
    } else if (type == rectangle) {
        adjustedCornerPoints = cornerPoints;
    }

    // Determine bounding box center of planeCloud
    Eigen::Vector3f minPoint = planeCloud->points[0].getVector3fMap();
    Eigen::Vector3f maxPoint = planeCloud->points[0].getVector3fMap();
    for (const auto& point : planeCloud->points) {
        Eigen::Vector3f pt = point.getVector3fMap();
        minPoint = minPoint.cwiseMin(pt);
        maxPoint = maxPoint.cwiseMax(pt);
    }

    Eigen::Vector3f boundingBoxCenter = (minPoint + maxPoint) / 2.0f;
    std::vector<Eigen::Vector3f> bestadjustedCornerPoints = adjustedCornerPoints;
    float initialScore = calculateBoundaryScore(planeCloud, bestadjustedCornerPoints, planeNormal);
    float bestScore = std::numeric_limits<float>::max();
    if (initialScore > 5.0){
        bestadjustedCornerPoints = applyTranslationToCorners(bestadjustedCornerPoints, boundingBoxCenter);
        for (int iter = 0; iter < maxIterations; ++iter) {

            #pragma omp parallel for shared(bestScore, bestadjustedCornerPoints)
            for (int i = 0; i < static_cast<int>(angleTotal / angleStep); ++i) {
                float angle = i * angleStep;

                // Apply both +angle and -angle rotations
                for (float angleDirection : {angle, -angle}) {
                    std::vector<Eigen::Vector3f> rotatedCorners = applyRotationToCorners(bestadjustedCornerPoints, boundingBoxCenter, planeNormal, angleDirection);
                    float currentScore = calculateBoundaryScore(planeCloud, rotatedCorners, planeNormal);

                    #pragma omp critical
                    {
                        if (currentScore < bestScore) {
                            bestScore = currentScore;
                            bestadjustedCornerPoints = rotatedCorners;
                        }
                    }
                }
            }

            bestadjustedCornerPoints = applyTranslationToCorners(bestadjustedCornerPoints, boundingBoxCenter);
        }
    }
    boundaryScore = bestScore;  // Set final boundary score
    return bestadjustedCornerPoints;
}