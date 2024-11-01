#pragma once

#include <cstdint>
#include <vector>

// Function declarations for initializing and cleaning up resources
// These are the functions for resource management and vehicle line detection
extern void CreateAswarmTriangleCallibration();  // Function to initialize resources (if needed)
extern void DeleteAswarmTriangleCallibration();  // Function to clean up resources (if needed)

// Main function to perform vehicle line detection
// Parameters:
// - leafSize: The size of the voxel grid for downsampling.
// - cluster_tolerance: The distance threshold for clustering.
// - min_cluster_size: The minimum number of points to form a valid cluster.
// - max_cluster_size: The maximum number of points for a cluster to be considered.
// - distance_threshold: The distance threshold for RANSAC plane detection.
// - inputCloud: The input point cloud (x, y, z) as a float array.
// - numInputCloud: The number of points in the input cloud.
// - outputVehicleLine: The output array to store the line features (x, y, z).
// - numoutputVehicleLine: The number of points in the line features output.
// - icp_uncertainty: A reference to a double to store the uncertainty from ICP alignment.

extern void OutputAswarmTriangleCallibration( float* inputCloud,                              uint32_t numInputCloud,
                                        double leafSize,                                                                                                                                        //Downsampling Parameter
                                        double clusterTolerance,                        uint32_t minClusterSize,                    uint32_t maxClusterSize,                                    //Vehicle Clustering Parameter
                                        double radiusSearch,                            double curvatureThresholdPlane,             double curvatureThresholdCorner,                            //Feature Extraction Parameter
                                        double normalDistanceMin,                       double normalDistanceMax,                                                                               //Plane Limit Parameter
                                        double adjacentSide,                            double oppositeSide,                        double hypotenuse,                                          //Ideal Triangle Parameter
                                        double adjacentAngle,                           double oppositeAngle,                       double vertexAngle,                                         //Ideal Triangle Parameter
                                        double distanceTolerance,                       double angleTolerance,                      double searchRadius,                                        //Candidate Search Parameter
                                        float* outputPlaneCloud,                        uint32_t& numoutputPlaneCloud,
                                        float* outputPcaTriangle,                       uint32_t& numoutputPcaTriangle,
                                        float* outputAdjustedPcaTriangle,               uint32_t& numoutputAdjustedPcaTriangle,
                                        float* outputBestTriangle,                      uint32_t& numoutputBestTriangle,
                                        float* outputCorrectedTriangle,                 uint32_t& numoutputCorrectedTriangle,
                                        float* outputAdjustedTriangle,                  uint32_t& numoutputAdjustedTriangle,
                                        float* outputFinalTriangle,                     uint32_t& numoutputFinalTriangle,
                                        float* outputoriginTriangle,                    uint32_t& numoutputOriginTriangle,
                                        float* outputTranslation,                       float* outputEulerAngles, 
                                        float& fitnessScore);

