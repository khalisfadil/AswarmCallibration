#pragma once

#include <cstdint>
// #include <cstddef>
// #include <cstdlib>  // For rand() and srand()
// #include <ctime>    // For time()

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
#include <cmath>  // For std::sqrt
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
    float a = coefficients->values[0];
    float b = coefficients->values[1];
    float c = coefficients->values[2];

    // Return the plane normal as an Eigen::Vector3f
    return Eigen::Vector3f(a, b, c);
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
    Eigen::Vector3f bestA, bestB, bestC;
    int bestBIndexSource = -1;  // -1 indicates no source identified

    // Step 1: Build the KdTree
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(candidatePointsCloud);

    // Step 2: Find candidate points for pointA, pointB, and pointC
    std::vector<Eigen::Vector3f> candidatesA = findCandidatePoints(originA, candidatePointsCloud, kdtree, searchRadius);
    std::vector<Eigen::Vector3f> candidatesB1 = findCandidatePoints(originB, candidatePointsCloud, kdtree, searchRadius);
    std::vector<Eigen::Vector3f> candidatesC = findCandidatePoints(originC, candidatePointsCloud, kdtree, searchRadius);
    std::vector<Eigen::Vector3f> candidatesB2 = findCandidatePoints(originD, candidatePointsCloud, kdtree, searchRadius);

    // Combine candidatesB1 and candidatesB2 into candidatesB, tracking the source for each point
    std::vector<Eigen::Vector3f> candidatesB = candidatesB1;
    std::vector<int> sourceIndexB(candidatesB1.size(), 1);  // 1 for candidatesB1

    candidatesB.insert(candidatesB.end(), candidatesB2.begin(), candidatesB2.end());
    sourceIndexB.insert(sourceIndexB.end(), candidatesB2.size(), 2);  // 2 for candidatesB2

    // Step 3: Evaluate triangles with OpenMP parallelization
    #pragma omp parallel
    {
        float localBestScore = std::numeric_limits<float>::max();
        Eigen::Vector3f localBestA, localBestB, localBestC;
        int localBestBIndexSource = -1;

        #pragma omp for
        for (uint32_t i = 0; i < candidatesB.size(); ++i) {
            const auto& candidateB = candidatesB[i];
            int currentBIndexSource = sourceIndexB[i];

            for (const auto& candidateA : candidatesA) {
                for (const auto& candidateC : candidatesC) {
                    float score = calculateTriangleScore(candidateA, candidateB, candidateC, 
                                                            distTolerance, angleTolerance,
                                                            adjacentSide, oppositeSide, hypotenuse, //float idealAB, float idealBC, float idealCA,
                                                            adjacentAngle, vertexAngle, oppositeAngle); // float idealAngleABBC, float idealAngleBCCA, float idealAngleCAAB

                    if (score < localBestScore) {
                        localBestScore = score;
                        localBestA = candidateA;
                        localBestB = candidateB;
                        localBestC = candidateC;
                        localBestBIndexSource = currentBIndexSource;
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (localBestScore < bestScore) {
                bestScore = localBestScore;
                bestA = localBestA;
                bestB = localBestB;
                bestC = localBestC;
                bestBIndexSource = localBestBIndexSource;
            }
        }
    }

    float dAB = calculateDistance(bestA, bestB);
    float dBC = calculateDistance(bestB, bestC);
    std::vector<Eigen::Vector3f> bestABC;

    if (dAB >= dBC){
        bestABC = { bestA, bestB, bestC };

    }else{
        bestABC = { bestC, bestB, bestA };
    }

    return std::make_tuple(bestABC, bestScore, bestBIndexSource);
}

// Function to recorrect triangle based on ideal distances and angles
std::vector<Eigen::Vector3f> recorrectTriangle(const std::vector<Eigen::Vector3f>& bestPoints, int indexSource,
                                                float idealAB, float idealBC, float idealCA,
                                                float idealAngleABBC, float idealAngleBCCA, float idealAngleCAAB) {
    const float tolerance = 0.0001f;  // Tolerance for distance and angle correction

    // Initialize corrected points with the current best points
    Eigen::Vector3f A = bestPoints[0];
    Eigen::Vector3f B = bestPoints[1];
    Eigen::Vector3f C = bestPoints[2];

    // Iteratively adjust until within tolerance
    int maxIterations = 100;
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
        if (std::abs(currentAB - idealAB) < tolerance &&
            std::abs(currentBC - idealBC) < tolerance &&
            std::abs(currentCA - idealCA) < tolerance &&
            std::abs(currentAngleABBC - idealAngleABBC) < tolerance &&
            std::abs(currentAngleBCCA - idealAngleBCCA) < tolerance &&
            std::abs(currentAngleCAAB - idealAngleCAAB) < tolerance) {
            break;  // Exit loop if all within tolerance
        }

        if (indexSource == 1){
            // Adjust points to match ideal distances
            if(std::abs(currentCA - idealCA) > tolerance){
                Eigen::Vector3f directionAC = (C - A).normalized();
                C = A + directionAC * idealCA;  // Adjust C along the line towards the ideal AC length
            }
            if(std::abs(currentBC - idealBC) > tolerance){
                Eigen::Vector3f directionCB = (B - C).normalized();
                B = C + directionCB * idealBC;  // Adjust B along the line towards the ideal CB length
            }
            if(std::abs(currentAB - idealAB) > tolerance){
                Eigen::Vector3f directionBA = (A - B).normalized();
                A = B + directionBA * idealAB;  // Adjust A along the line towards the ideal BA length
            }
        }else{
             // Adjust points to match ideal distances
            if (std::abs(currentCA - idealCA) > tolerance) {
                Eigen::Vector3f directionCA = (A - C).normalized();
                A = C + directionCA * idealCA;  // Adjust A along the line towards the ideal CA length
            }
            if (std::abs(currentAB - idealAB) > tolerance) {
                Eigen::Vector3f directionAB = (B - A).normalized();
                B = A + directionAB * idealAB;  // Adjust B along the line towards the ideal AB length
            }
            if (std::abs(currentBC - idealBC) > tolerance) {
                Eigen::Vector3f directionBC = (C - B).normalized();
                C = B + directionBC * idealBC;  // Adjust C along the line towards the ideal BC length
            }
        }
    }

    // Return the corrected points
    return { A, B, C };
}

// Function to adjust triangle points to create a rectangular boundary that encompasses the plane cloud
std::vector<Eigen::Vector3f> readjustCornerPoints(const std::vector<Eigen::Vector3f>& cornerPoints,
                                                  pcl::PointCloud<pcl::PointXYZ>::Ptr planeCloud, 
                                                  uint32_t type = 1) {
    // Constants
    constexpr uint32_t triangle = 1;
    constexpr uint32_t rectangle = 2;
    std::vector<Eigen::Vector3f> adjustedCornerPoints;

    // Step 1: Define corners based on type
    if (type == triangle) {
        // Calculate the fourth corner (D) to form a rectangle
        Eigen::Vector3f A = cornerPoints[0];
        Eigen::Vector3f B = cornerPoints[1];
        Eigen::Vector3f C = cornerPoints[2];
        
        Eigen::Vector3f AB = (B - A).normalized();
        Eigen::Vector3f AC = (C - A).normalized();
        Eigen::Vector3f AD = AB.cross(AC).cross(AB) * (C - B).norm();
        Eigen::Vector3f D = A + AD;

        adjustedCornerPoints = {A, B, C, D}; // Rectangle from triangle points

    } else if (type == rectangle) {
        // Rectangle points directly provided
        adjustedCornerPoints = cornerPoints;
    }

    // Step 2: Calculate the initial center of the rectangle
    Eigen::Vector3f rectangleCenter = (adjustedCornerPoints[0] + adjustedCornerPoints[1] + 
                                       adjustedCornerPoints[2] + adjustedCornerPoints[3]) / 4.0f;

    // Step 3: Determine bounding box for planeCloud
    Eigen::Vector3f minPoint = planeCloud->points[0].getVector3fMap();
    Eigen::Vector3f maxPoint = planeCloud->points[0].getVector3fMap();
    for (const auto& point : planeCloud->points) {
        Eigen::Vector3f pt = point.getVector3fMap();
        minPoint = minPoint.cwiseMin(pt);
        maxPoint = maxPoint.cwiseMax(pt);
    }

    // Step 4: Calculate adaptive offset based on bounding box and rectangle center
    Eigen::Vector3f boundingBoxCenter = (minPoint + maxPoint) / 2.0f;
    Eigen::Vector3f totalOffset = (boundingBoxCenter - rectangleCenter) * 0.5f;  // Adaptive scaling factor

    // Step 5: Apply offset to each corner and return adjusted points
    for (auto& corner : adjustedCornerPoints) {
        corner += totalOffset;
    }

    return adjustedCornerPoints;
}

