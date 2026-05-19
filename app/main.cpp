#include <filesystem>
#include <iostream>
#include <vector>
#include "utils/Config.h"
#include "imageProcess/ImageProcess.h"
#include "imageProcess/CannyDetecter.h"

int main() {
   
    camcalib::ImageProcess process;
    process.runCalibrate();


    return 0;
}
