#include <string>
#include <vector>



namespace camcalib{
    
struct CaliConfig {
    int board_width;      // 棋盘格内角点列数 (例如 9)
    int board_height;     // 棋盘格内角点行数 (例如 6)
    double square_size;   // 棋盘格单格物理尺寸 (单位：mm 或 m)
    std::string image_dir;  // 标定图像所在目录
};


class ConfigReader {

    public:
    static bool readConfig(const std::string& yaml_path, CaliConfig& config);

    static std::vector<std::string> getImageFiles(const std::string& dir);
};


}


