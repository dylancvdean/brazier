#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

struct sd_ctx_t;

namespace brazier {

struct ImageRequest {
    std::string prompt;
    std::string negative;
    int width = 512;
    int height = 512;
    int steps = 0;      // 0 keeps the library default
    double cfg = -1;    // negative keeps the library default
    int64_t seed = -1;  // negative picks a random seed
};

// Row-major interleaved RGB.
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> pixels;
};

enum class Outcome { Done, Cancelled, Failed };

using Progress = std::function<void(const std::string& stage, int64_t completed, int64_t total)>;

// One stable-diffusion.cpp model at a time. load, generate and unload must not
// overlap each other; cancel may be called from any thread.
class ImageEngine {
public:
    ~ImageEngine();
    bool load(const std::string& model_path, bool gpu, int threads, std::string& error);
    Outcome generate(const ImageRequest& request, const Progress& progress, Image& image, std::string& error);
    void cancel();
    void unload();

private:
    std::mutex cancel_mutex_;
    sd_ctx_t* ctx_ = nullptr;
    Progress progress_;
};

}  // namespace brazier
