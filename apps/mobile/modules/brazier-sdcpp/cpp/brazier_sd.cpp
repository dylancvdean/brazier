#include "brazier_sd.h"

#include "stable-diffusion.h"

namespace brazier {

namespace {

std::string last_error(const char* fallback) {
    const char* text = sd_get_last_error();
    return text && *text ? text : fallback;
}

struct Options {
    sd_opt_t* value;
    ~Options() { sd_opt_free(value); }
};

}  // namespace

ImageEngine::~ImageEngine() {
    unload();
}

bool ImageEngine::load(const std::string& model_path, bool gpu, int threads, std::string& error) {
    unload();
    Options options{sd_ctx_opt_create()};
    sd_ctx_t* ctx = nullptr;
    // Weights stay memory-mapped so that they are file-backed pages the system
    // can evict, instead of counting against the app's memory limit.
    if (!options.value ||
        sd_opt_set_string(options.value, "model_path", model_path.c_str()) != SD_STATUS_OK ||
        sd_opt_set_bool(options.value, "enable_mmap", true) != SD_STATUS_OK ||
        (threads > 0 && sd_opt_set_int(options.value, "n_threads", threads) != SD_STATUS_OK) ||
        (!gpu && sd_opt_set_string(options.value, "backend", "cpu") != SD_STATUS_OK) ||
        sd_ctx_create(options.value, &ctx) != SD_STATUS_OK) {
        error = last_error("Could not open the model.");
        return false;
    }
    if (!sd_ctx_supports_image_generation(ctx)) {
        error = "This file is not an image model that stable-diffusion.cpp can run.";
        sd_ctx_free(ctx);
        return false;
    }
    sd_set_generation_progress_callback(
        ctx,
        [](const sd_generation_progress_t* event, void* data) {
            auto* self = static_cast<ImageEngine*>(data);
            if (self->progress_) {
                self->progress_(event->stage ? event->stage : "", event->completed, event->total);
            }
        },
        this);
    std::lock_guard<std::mutex> lock(cancel_mutex_);
    ctx_ = ctx;
    return true;
}

Outcome ImageEngine::generate(const ImageRequest& request, const Progress& progress, Image& image, std::string& error) {
    if (!ctx_) {
        error = "No image model is loaded.";
        return Outcome::Failed;
    }
    Options options{sd_img_gen_opt_create()};
    bool ok = options.value &&
              sd_opt_set_string(options.value, "prompt", request.prompt.c_str()) == SD_STATUS_OK &&
              sd_opt_set_string(options.value, "negative_prompt", request.negative.c_str()) == SD_STATUS_OK &&
              sd_opt_set_int(options.value, "width", request.width) == SD_STATUS_OK &&
              sd_opt_set_int(options.value, "height", request.height) == SD_STATUS_OK &&
              sd_opt_set_int(options.value, "seed", request.seed) == SD_STATUS_OK &&
              // Decoding in tiles keeps the VAE's working memory within a phone's limit.
              sd_opt_set_bool(options.value, "vae_tiling.enabled", true) == SD_STATUS_OK &&
              (request.steps <= 0 || sd_opt_set_int(options.value, "sample.steps", request.steps) == SD_STATUS_OK) &&
              (request.cfg < 0 ||
               sd_opt_set_float(options.value, "sample.guidance.txt_cfg", request.cfg) == SD_STATUS_OK);
    if (!ok) {
        error = last_error("Those image settings are not valid.");
        return Outcome::Failed;
    }
    progress_ = progress;
    sd_cancel_generation(ctx_, SD_CANCEL_RESET);
    sd_result_t* result = nullptr;
    sd_status_t status = sd_generate_image(ctx_, options.value, &result);
    progress_ = nullptr;
    if (status == SD_STATUS_CANCELLED) {
        sd_result_free(result);
        return Outcome::Cancelled;
    }
    const sd_tensor_t* tensor =
        status == SD_STATUS_OK && result && sd_result_get_image_count(result) > 0 ? sd_result_get_image(result, 0) : nullptr;
    if (!tensor || sd_tensor_type(tensor) != SD_TENSOR_U8 || sd_tensor_rank(tensor) != 3) {
        error = last_error("Image generation failed.");
        sd_result_free(result);
        return Outcome::Failed;
    }
    image.channels = static_cast<int>(sd_tensor_dim(tensor, 0));
    image.width = static_cast<int>(sd_tensor_dim(tensor, 1));
    image.height = static_cast<int>(sd_tensor_dim(tensor, 2));
    const auto* data = static_cast<const uint8_t*>(sd_tensor_data(tensor));
    image.pixels.assign(data, data + sd_tensor_byte_size(tensor));
    sd_result_free(result);
    return Outcome::Done;
}

void ImageEngine::cancel() {
    std::lock_guard<std::mutex> lock(cancel_mutex_);
    if (ctx_) sd_cancel_generation(ctx_, SD_CANCEL_ALL);
}

void ImageEngine::unload() {
    sd_ctx_t* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(cancel_mutex_);
        ctx = ctx_;
        ctx_ = nullptr;
    }
    sd_ctx_free(ctx);
}

}  // namespace brazier
