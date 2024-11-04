#include "AswarmTriangleCallibrationHelper.h"
#include "AswarmTriangleCallibration.h"
// static std::deque<Eigen::Matrix<float,6,1>> transformation_memory;
// static const int memory_size = 5;

// Initialize resources
void CreateAswarmTriangleCallibration() {
    // Additional resources can be initialized here if needed
}

// Clean up resources
void DeleteAswarmTriangleCallibration() {
    // Clean up resources here if needed
}

// Main function to match the detected point cloud with the premade outline
void OutputAswarmTriangleCallibration( float* inputCloud,                              uint32_t numInputCloud,
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
                                        float& fitnessScore) 
{   
    // Initialized constant parameter
    float point_spacing = 0.1f;
    uint32_t triangle = 1;
    uint32_t rectangle = 2;
    uint32_t size = 256000;
    float boundaryScore;

    // Initialize output arrays with NaN values
    std::fill(outputPlaneCloud, outputPlaneCloud + size * 3, std::numeric_limits<float>::quiet_NaN());
    std::fill(outputPcaTriangle, outputPcaTriangle + size * 3, std::numeric_limits<float>::quiet_NaN());
    std::fill(outputAdjustedPcaTriangle, outputAdjustedPcaTriangle + size * 3, std::numeric_limits<float>::quiet_NaN());
    std::fill(outputBestTriangle, outputBestTriangle + size * 3, std::numeric_limits<float>::quiet_NaN());
    std::fill(outputCorrectedTriangle, outputCorrectedTriangle + size * 3, std::numeric_limits<float>::quiet_NaN());
    std::fill(outputAdjustedTriangle, outputAdjustedTriangle + size * 3, std::numeric_limits<float>::quiet_NaN());
    std::fill(outputFinalTriangle, outputFinalTriangle + size * 3, std::numeric_limits<float>::quiet_NaN());
    std::fill(outputoriginTriangle, outputoriginTriangle + size * 3, std::numeric_limits<float>::quiet_NaN());
    std::fill(outputTranslation, outputTranslation + 3, std::numeric_limits<float>::quiet_NaN());
    std::fill(outputEulerAngles, outputEulerAngles + 3, std::numeric_limits<float>::quiet_NaN());  
    fitnessScore = std::numeric_limits<float>::quiet_NaN();

    numoutputPlaneCloud = static_cast<uint32_t>(0);
    numoutputPcaTriangle = static_cast<uint32_t>(0);
    numoutputAdjustedPcaTriangle = static_cast<uint32_t>(0);
    numoutputBestTriangle = static_cast<uint32_t>(0);
    numoutputCorrectedTriangle = static_cast<uint32_t>(0);
    numoutputAdjustedTriangle = static_cast<uint32_t>(0);
    numoutputFinalTriangle = static_cast<uint32_t>(0);
    numoutputOriginTriangle = static_cast<uint32_t>(0);

    // ############################################################################
    // Step 1: Convert the float array to a PCL point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = convertToPointCloud(inputCloud, numInputCloud, size);

    // ############################################################################
    // Step 2: Downsample the point cloud using VoxelGrid filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud = downsamplePointCloud(cloud, leafSize); 

    // ############################################################################
    // Step3: return the nearest cluster using Euclidean Cluster Extraction
    float tolerance = static_cast<float>(clusterTolerance);
    pcl::PointCloud<pcl::PointXYZ>::Ptr vehicleCloud = findNearestCluster(filteredCloud, tolerance, minClusterSize, maxClusterSize);

    if (filteredCloud->empty()) {
        std::cout << "filteredCloud is empty. Exiting processing." << std::endl;
        return;  // Exit the function if no point cloud is present
    }

    // ############################################################################
    // Step 4: extract planar and corner features based on normal estimation
    auto featurePair = extractFeatures(vehicleCloud, radiusSearch, curvatureThresholdPlane, curvatureThresholdCorner);

    // Extract the planar and corner feature clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr planeFeatures = featurePair.first;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cornerFeatures = featurePair.second;

    if (planeFeatures->empty()) {
        std::cout << "planeFeatures is empty. Exiting processing." << std::endl;
        return;  // Exit the function if no point cloud is present
    }

    // ############################################################################
    // Step 5: find the plane normal coefficients by using RANSAC
    Eigen::Vector3f planeNormal = findPlaneCoefficients(planeFeatures);

    Eigen::Vector3f vehicleZaxis(0.0, 0.0, 1.0);

    // Compute the rotation matrix
    Eigen::Quaternionf rotation;
    rotation.setFromTwoVectors(planeNormal, vehicleZaxis);

    // ############################################################################
    // Step 6: Transform point cloud to align with vehicle coordinate system
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*vehicleCloud, *transformedCloud, Eigen::Affine3f(rotation));

    // ############################################################################
    // Step 7: Apply PassThrough filter on the transformed cloud in the vehicle's Z-axis
    float Min = static_cast<float>(normalDistanceMin);
    float Max = static_cast<float>(normalDistanceMax);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(transformedCloud);
    pass.setFilterFieldName("z");  
    pass.setFilterLimits(Min, Max);  // Filter out points above the vehicle top
    pass.filter(*transformedCloud);

    // ############################################################################
    // Step 8: Transform the point cloud back to the original Lidar coordinates
    Eigen::Quaternionf inverse_rotation = rotation.inverse();  // Compute the inverse rotation
    pcl::transformPointCloud(*transformedCloud, *transformedCloud, Eigen::Affine3f(inverse_rotation));

    if (transformedCloud->empty()) {
        std::cout << "transformedCloud is empty. Exiting processing." << std::endl;
        return;  // Exit the function if no point cloud is present
    } else {
        // Output of the detected plane
        uint32_t planeSize = transformedCloud->points.size();
        #pragma omp prallel for
        for (uint32_t i = 0; i < planeSize; ++i){
            outputPlaneCloud[i] = transformedCloud->points[i].x;
            outputPlaneCloud[i + size] = transformedCloud->points[i].y;
            outputPlaneCloud[i + size * 2] = transformedCloud->points[i].z;
        }
        numoutputPlaneCloud = static_cast<uint32_t>(planeSize);
    }
    
    // ############################################################################
    // Step 9: Find the corners of a known-dimension rectangular plane using PCA
    std::vector<Eigen::Vector3f> pcaCorners = findRectangleCornersWithPCA(transformedCloud, adjacentSide, oppositeSide);

    // create a premade outline to vizualize
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcaTriangle(new pcl::PointCloud<pcl::PointXYZ>);

    // Generate and combine the points for each edge of the rectangle
    *pcaTriangle += *interpolatePoints(pcaCorners[0], pcaCorners[1], point_spacing); 
    *pcaTriangle += *interpolatePoints(pcaCorners[1], pcaCorners[2], point_spacing); 
    *pcaTriangle += *interpolatePoints(pcaCorners[2], pcaCorners[0], point_spacing);

    *pcaTriangle += *interpolatePoints(pcaCorners[3], pcaCorners[2], point_spacing); 
    *pcaTriangle += *interpolatePoints(pcaCorners[2], pcaCorners[1], point_spacing);  
    *pcaTriangle += *interpolatePoints(pcaCorners[1], pcaCorners[3], point_spacing);

    *pcaTriangle += *interpolatePoints(pcaCorners[1], pcaCorners[0], point_spacing); 
    *pcaTriangle += *interpolatePoints(pcaCorners[0], pcaCorners[3], point_spacing);  
    *pcaTriangle += *interpolatePoints(pcaCorners[3], pcaCorners[1], point_spacing);

    *pcaTriangle += *interpolatePoints(pcaCorners[2], pcaCorners[3], point_spacing); 
    *pcaTriangle += *interpolatePoints(pcaCorners[3], pcaCorners[0], point_spacing); 
    *pcaTriangle += *interpolatePoints(pcaCorners[0], pcaCorners[2], point_spacing);

    if (pcaTriangle->empty()) {
        std::cout << "pcaTriangle are empty. Exiting processing." << std::endl;
        return;
    }else {
        // Output the bestTriangle
        uint32_t pcaTriangleSize = pcaTriangle->points.size();
        #pragma omp parallel for
        for (uint32_t i = 0; i < pcaTriangleSize; ++i) {
            outputPcaTriangle[i] = pcaTriangle->points[i].x;
            outputPcaTriangle[i + size] = pcaTriangle->points[i].y;
            outputPcaTriangle[i + size * 2] = pcaTriangle->points[i].z;
        }
        numoutputPcaTriangle = static_cast<uint32_t>(pcaTriangleSize);
    }

    // ############################################################################
    // Step 10: adjust the corners points to create a rectangular boundary that encompasses the plane cloud
    std::vector<Eigen::Vector3f> adjustedPcaCorner = readjustCornerPoints(pcaCorners, transformedCloud, planeNormal, rectangle, boundaryScore, 0.1, 3.0, 3);

    // create a premade outline to vizualize
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjustedPcaTriangle(new pcl::PointCloud<pcl::PointXYZ>);

    // Generate and combine the points for each edge of the rectangle
    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[0], adjustedPcaCorner[1], point_spacing);
    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[1], adjustedPcaCorner[2], point_spacing);
    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[2], adjustedPcaCorner[0], point_spacing);

    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[3], adjustedPcaCorner[2], point_spacing);
    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[2], adjustedPcaCorner[1], point_spacing);
    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[1], adjustedPcaCorner[3], point_spacing);

    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[1], adjustedPcaCorner[0], point_spacing); 
    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[0], adjustedPcaCorner[3], point_spacing);  
    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[3], adjustedPcaCorner[1], point_spacing);

    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[2], adjustedPcaCorner[3], point_spacing); 
    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[3], adjustedPcaCorner[0], point_spacing); 
    *adjustedPcaTriangle += *interpolatePoints(adjustedPcaCorner[0], adjustedPcaCorner[2], point_spacing);

    if (adjustedPcaTriangle->empty()) {
        std::cout << "adjustedPcaTriangle are empty. Exiting processing." << std::endl;
        return;
    }else {
        // Output the bestTriangle
        uint32_t adjustedPcaTriangleSize = adjustedPcaTriangle->points.size();
        #pragma omp parallel for
        for (uint32_t i = 0; i < adjustedPcaTriangleSize; ++i) {
            outputAdjustedPcaTriangle[i] = adjustedPcaTriangle->points[i].x;
            outputAdjustedPcaTriangle[i + size] = adjustedPcaTriangle->points[i].y;
            outputAdjustedPcaTriangle[i + size * 2] = adjustedPcaTriangle->points[i].z;
        }
        numoutputAdjustedPcaTriangle = static_cast<uint32_t>(adjustedPcaTriangleSize);
    }

    // ############################################################################
    // // Step 11: 
    // Eigen::Vector3f originA = adjustedPcaCorner[0]; // TOP LEFT
    // Eigen::Vector3f originB = adjustedPcaCorner[1]; // BOTTOM LEFT
    // Eigen::Vector3f originC = adjustedPcaCorner[2]; // BOTTOM RIGHT
    // Eigen::Vector3f originD = adjustedPcaCorner[3]; // TOP RIGHT
    
    // ############################################################################
    // Step 11: Function to find the best triangle and return best points and score
    auto result = findBestTriangle(adjustedPcaCorner[0], adjustedPcaCorner[1], adjustedPcaCorner[2], adjustedPcaCorner[3], transformedCloud, 
                                    searchRadius, distanceTolerance, angleTolerance,
                                    adjacentSide, oppositeSide, hypotenuse,
                                    adjacentAngle, vertexAngle, oppositeAngle);

    // Extract the best points, score, and index source from the tuple
    std::vector<Eigen::Vector3f> bestPoints = std::get<0>(result);
    float bestScore = std::get<1>(result);
    int indexSource = std::get<2>(result); 

    // create a premade outline to vizualize
    pcl::PointCloud<pcl::PointXYZ>::Ptr bestTriangle(new pcl::PointCloud<pcl::PointXYZ>);

    // Generate and combine the points for each edge of the rectangle
    *bestTriangle += *interpolatePoints(bestPoints[0], bestPoints[1], point_spacing);
    *bestTriangle += *interpolatePoints(bestPoints[1], bestPoints[2], point_spacing);
    *bestTriangle += *interpolatePoints(bestPoints[2], bestPoints[0], point_spacing);
    
    if (bestTriangle->empty()) {
        std::cout << "bestTriangle is empty. Exiting processing." << std::endl;
        return;  // Exit the function if no point cloud is present
    } else {
        // Output the bestTriangle
        uint32_t bestTriangleSize = bestTriangle->points.size();
        #pragma omp parallel for
        for (uint32_t i = 0; i < bestTriangleSize; ++i) {
            outputBestTriangle[i] = bestTriangle->points[i].x;
            outputBestTriangle[i + size] = bestTriangle->points[i].y;
            outputBestTriangle[i + size * 2] = bestTriangle->points[i].z;
        }
        numoutputBestTriangle = static_cast<uint32_t>(bestTriangleSize);
    }

    // ############################################################################
    // Step 12: recorrect triangle based on ideal distances and angles
    std::vector<Eigen::Vector3f> correctedPoints = recorrectTriangle(bestPoints, indexSource,
                                                                    adjacentSide, oppositeSide, hypotenuse,
                                                                    adjacentAngle, vertexAngle, oppositeAngle);

    // create a premade outline to vizualize
    pcl::PointCloud<pcl::PointXYZ>::Ptr correctedTriangle(new pcl::PointCloud<pcl::PointXYZ>);

    // Generate and combine the points for each edge of the rectangle
    *correctedTriangle += *interpolatePoints(correctedPoints[0], correctedPoints[1], point_spacing);
    *correctedTriangle += *interpolatePoints(correctedPoints[1], correctedPoints[2], point_spacing);
    *correctedTriangle += *interpolatePoints(correctedPoints[2], correctedPoints[0], point_spacing);

    if (correctedTriangle->empty()) {
        std::cout << "correctedTriangle is empty. Exiting processing." << std::endl;
        return;  // Exit the function if no point cloud is present
    }else {
        // Output the correctedTriangle
        uint32_t correctedTriangleSize = correctedTriangle->points.size();
        #pragma omp prallel for
        for (uint32_t i = 0; i < correctedTriangleSize; ++i){
            outputCorrectedTriangle[i] = correctedTriangle->points[i].x;
            outputCorrectedTriangle[i + size] = correctedTriangle->points[i].y;
            outputCorrectedTriangle[i + size * 2] = correctedTriangle->points[i].z;
        }
        numoutputCorrectedTriangle = static_cast<uint32_t>(correctedTriangleSize);
    }

    // ############################################################################
    // Step 13: adjust triangle points to create a rectangular boundary that encompasses the plane cloud
    std::vector<Eigen::Vector3f> adjustedPoints = readjustCornerPoints(correctedPoints, transformedCloud, planeNormal, triangle, boundaryScore, 0.05, 0.01, 1);

    // create a premade outline to vizualize
    pcl::PointCloud<pcl::PointXYZ>::Ptr adjustedTriangle(new pcl::PointCloud<pcl::PointXYZ>);

    // Generate and combine the points for each edge of the rectangle
    *adjustedTriangle += *interpolatePoints(adjustedPoints[0], adjustedPoints[1], point_spacing);
    *adjustedTriangle += *interpolatePoints(adjustedPoints[1], adjustedPoints[2], point_spacing);
    *adjustedTriangle += *interpolatePoints(adjustedPoints[2], adjustedPoints[0], point_spacing);

    if (adjustedTriangle->empty()) {
        std::cout << "adjustedTriangle is empty. Exiting processing." << std::endl;
        return;  // Exit the function if no point cloud is present
    }else {
         // Output the adjustedTriangle
        uint32_t adjustedTriangleSize = adjustedTriangle->points.size();
        #pragma omp prallel for
        for (uint32_t i = 0; i < adjustedTriangleSize; ++i){
            outputAdjustedTriangle[i] = adjustedTriangle->points[i].x;
            outputAdjustedTriangle[i + size] = adjustedTriangle->points[i].y;
            outputAdjustedTriangle[i + size * 2] = adjustedTriangle->points[i].z;
        }
        numoutputAdjustedTriangle = static_cast<uint32_t>(adjustedTriangleSize);
    }

    // ############################################################################
    // // Step 14: Offset premadeL by -1 in the z-direction of the plane's normal
    // Eigen::Vector3f offset = -1.0 * planeNormal.normalized();  // Offset by -1 unit along the plane's normal

    // for (auto& point : premadeL->points) {
    //     point.x += offset.x();
    //     point.y += offset.y();
    //     point.z += offset.z();
    // }

    // ############################################################################
    // Step 15: Create Premade Outline
    // Define the four corners of the rectangle, with the surface facing the Z-direction
    Eigen::Vector3f corners[4];
    corners[0] = Eigen::Vector3f(adjacentSide / 2.0f, oppositeSide / 2.0f, 0);    
    corners[1] = Eigen::Vector3f(-adjacentSide / 2.0f, oppositeSide / 2.0f, 0);  
    corners[2] = Eigen::Vector3f(-adjacentSide / 2.0f, -oppositeSide / 2.0f, 0);  
    corners[3] = Eigen::Vector3f(adjacentSide / 2.0f, -oppositeSide / 2.0f, 0);   
     
    pcl::PointCloud<pcl::PointXYZ>::Ptr originTriangle(new pcl::PointCloud<pcl::PointXYZ>);

    if (indexSource == 1) {
        // If indexSource is 1, interpolate between corners[0] and corners[1]
        *originTriangle += *interpolatePoints(corners[0], corners[1], point_spacing);
        *originTriangle += *interpolatePoints(corners[1], corners[2], point_spacing);
        *originTriangle += *interpolatePoints(corners[2], corners[0], point_spacing);
    } else if (indexSource == 2) {
        // If indexSource is 2, interpolate between corners[0] and corners[3]
        *originTriangle += *interpolatePoints(corners[1], corners[2], point_spacing);
        *originTriangle += *interpolatePoints(corners[2], corners[3], point_spacing);
        *originTriangle += *interpolatePoints(corners[3], corners[1], point_spacing);
    } else if (indexSource == 3) {
        // If indexSource is 2, interpolate between corners[0] and corners[3]
        *originTriangle += *interpolatePoints(corners[2], corners[3], point_spacing);
        *originTriangle += *interpolatePoints(corners[3], corners[0], point_spacing);
        *originTriangle += *interpolatePoints(corners[0], corners[2], point_spacing);
    } else if (indexSource == 4) {
        // If indexSource is 2, interpolate between corners[0] and corners[3]
        *originTriangle += *interpolatePoints(corners[3], corners[0], point_spacing);
        *originTriangle += *interpolatePoints(corners[0], corners[1], point_spacing);
        *originTriangle += *interpolatePoints(corners[1], corners[3], point_spacing);
    }
    

    if (originTriangle->empty()) {
        std::cout << "originTriangle is empty. Exiting processing." << std::endl;
        return;  // Exit the function if no point cloud is present
    }else{
        // Output the premadeTOrigin
        uint32_t originTriangleSize = originTriangle->points.size();
        #pragma omp prallel for
        for (uint32_t i = 0; i < originTriangleSize; ++i){
            outputoriginTriangle[i] = originTriangle->points[i].x;
            outputoriginTriangle[i + size] = originTriangle->points[i].y;
            outputoriginTriangle[i + size * 2] = originTriangle->points[i].z;
        }
        numoutputOriginTriangle = static_cast<uint32_t>(originTriangleSize);
    }

    float threshold = 0.001f;  // Fitness score threshold
    int maxIterations = 50;    // Maximum number of iterations
    float score = std::numeric_limits<float>::max();  // Initialize to a high value
    Eigen::Matrix4f cumulativeTransformation = Eigen::Matrix4f::Identity();  // Initialize cumulative transformation

    // Copy of the adjusted triangle for iterative updates
    pcl::PointCloud<pcl::PointXYZ>::Ptr finalTriangle(new pcl::PointCloud<pcl::PointXYZ>(*adjustedTriangle));

    // Iterative scan matching with max iterations and fitness threshold
    int iteration = 0;
    while (score > threshold && iteration < maxIterations) {
        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
        gicp.setInputSource(finalTriangle);  
        gicp.setInputTarget(originTriangle);      

        // Align and get fitness score for this iteration
        gicp.align(*finalTriangle);  
        Eigen::Matrix4f currentTransformation = gicp.getFinalTransformation();
        score = gicp.getFitnessScore();

        // Update cumulative transformation
        cumulativeTransformation = cumulativeTransformation * currentTransformation;

        // Increment iteration counter
        iteration++;
    }
    
    fitnessScore = score;

    if (finalTriangle->empty()) {
        std::cout << "finalTriangle is empty. Exiting processing." << std::endl;
        return;  // Exit the function if no point cloud is present
    }else{
        // Output the finalAligned
        uint32_t finalTriangleSize = finalTriangle->points.size();
        #pragma omp prallel for
        for (uint32_t i = 0; i < finalTriangleSize; ++i){
            outputFinalTriangle[i] = finalTriangle->points[i].x;
            outputFinalTriangle[i + size] = finalTriangle->points[i].y;
            outputFinalTriangle[i + size * 2] = finalTriangle->points[i].z;
        }
        numoutputFinalTriangle = static_cast<uint32_t>(finalTriangleSize);
    }

    // ############################################################################
    // Step 17: define output
    Eigen::Vector3f translation;
    Eigen::Vector3f eulerAngles;
    Eigen::Affine3f affineTrans(cumulativeTransformation);
    float roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(affineTrans, translation[0], translation[1], translation[2], roll, pitch, yaw);
    eulerAngles = Eigen::Vector3f(roll, pitch, yaw);

    // Copy the translation and Euler angles to the output arrays
    outputTranslation[0] = translation[0];
    outputTranslation[1] = translation[1];
    outputTranslation[2] = translation[2];

    outputEulerAngles[0] = roll;
    outputEulerAngles[1] = pitch;
    outputEulerAngles[2] = yaw;

}










