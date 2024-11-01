defs = [];
directorySource = '';  % Path to the source files

% Driver: Info Request Encode
def = legacy_code('initialize');
def.SFunctionName = 'SFunction_AswarmTriangleCallibration';
def.StartFcnSpec  = 'void CreateAswarmTriangleCallibration()';
def.OutputFcnSpec = ['void OutputAswarmTriangleCallibration(single u1[256000][3], uint32 u2,' ...  % Cloud Input
                     'double u3,' ...                                                       % Downsampling Parameter
                     'double u4, uint32 u5, uint32 u6,' ...                                 % Vehicle Clustering Parameter
                     'double u7, double u8, double u9,' ...                                 % Feature Extraction Parameter
                     'double u10, double u11,' ...                                          % Plane Limit Parameter
                     'double u12, double u13, double u14,' ...                              % Ideal Triangle Parameter
                     'double u15, double u16, double u17,' ...                              % Ideal Triangle Parameter
                     'double u18, double u19, double u20,' ...                              % Candidate Search Parameter
                     'single y1[256000][3], uint32 y2,' ...                                 % outputPlaneCloud
                     'single y3[256000][3], uint32 y4,' ...                                 % outputPcaTriangle
                     'single y5[256000][3], uint32 y6,' ...                                 % outputAdjustedPcaTriangle
                     'single y7[256000][3], uint32 y8,' ...                                 % outputBestTriangle
                     'single y9[256000][3], uint32 y10,' ...                                % outputCorrectedTriangle
                     'single y11[256000][3], uint32 y12,' ...                               % outputAdjustedTriangle
                     'single y13[256000][3], uint32 y14,' ...                               % outputFinalTriangle
                     'single y15[256000][3], uint32 y16,' ...                               % outputoriginTriangle
                     'single y17[3][1], single y18[3][1],' ...                              % output Transformation
                     'single y19)'];                                                        % output GICP fittness score
def.TerminateFcnSpec = 'void DeleteAswarmTriangleCallibration()';
def.HeaderFiles   = {'AswarmTriangleCallibration.h'};
def.SourceFiles   = {'AswarmTriangleCallibration.cpp'};
def.IncPaths      = {directorySource};   % Adding source directory to include path
def.SrcPaths      = {directorySource};
def.Options.language = 'C++';
def.Options.useTlcWithAccel = false;   % Change to true if needed
def.SampleTime = 'parameterized';      % Adjust based on your design
defs = [defs; def];

% Compile and generate all required files
legacy_code('sfcn_cmex_generate', defs);

% Define paths and libraries based on the OS
if(ispc())
    includes = {''};
    libraries = {''};
elseif(isunix())
    includes = { ...
        '-I/usr/include/pcl-1.12', ...
        '-I/usr/include/eigen3', ...
        '-I/usr/include/vtk-9.1', ...
        ['-I' directorySource '/library/include']  % Adjust path if necessary
    };
    
    libraries = {
        '-L/usr/lib/x86_64-linux-gnu', ...
        '-lpcl_common', '-lpcl_io', '-lpcl_filters', '-lpcl_kdtree', ...
        '-lpcl_search', '-lpcl_features', '-lpcl_surface', '-lpcl_sample_consensus', ...
        '-lpcl_octree', '-lpcl_visualization', '-lpcl_segmentation', ...
        '-lvtkCommonCore-9.1', '-lboost_system', '-lboost_filesystem'
    };

    mexOpenMPFlags = {
        'CXXFLAGS="\$CXXFLAGS -fopenmp"', ...
        'LDFLAGS="\$LDFLAGS -fopenmp"'
    };
else
    includes = {''};
    libraries = {''};
end

% Compile using legacy_code
legacy_code('compile', defs, {includes{:}, libraries{:}, mexOpenMPFlags{:}});

% Generate TLC and Simulink blocks
legacy_code('sfcn_tlc_generate', defs);
legacy_code('rtwmakecfg_generate', defs);
legacy_code('slblock_generate', defs);

% Clean up variables
clear def defs directorySource includes libraries mexOpenMPFlags;
