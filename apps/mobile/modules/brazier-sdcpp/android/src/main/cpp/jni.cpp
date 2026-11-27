#include <jni.h>

#include <string>

#include "brazier_sd.h"

namespace {

brazier::ImageEngine engine;

std::string text(JNIEnv* env, jbyteArray value) {
    if (!value) return "";
    std::string result(static_cast<size_t>(env->GetArrayLength(value)), '\0');
    env->GetByteArrayRegion(value, 0, static_cast<jsize>(result.size()), reinterpret_cast<jbyte*>(&result[0]));
    return result;
}

void fail(JNIEnv* env, const char* type, const std::string& message) {
    env->ThrowNew(env->FindClass(type), message.c_str());
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_ai_brazier_sdcpp_BrazierSdcppModule_nativeLoad(JNIEnv* env, jobject, jbyteArray path, jint threads) {
    std::string error;
    // CPU only: stable-diffusion.cpp's GPU backends are not built for Android here.
    if (!engine.load(text(env, path), false, threads, error)) {
        fail(env, "java/lang/RuntimeException", error);
    }
}

// Returns ARGB pixels for android.graphics.Bitmap, or null when cancelled.
JNIEXPORT jintArray JNICALL
Java_ai_brazier_sdcpp_BrazierSdcppModule_nativeGenerate(
    JNIEnv* env, jobject self, jbyteArray prompt, jbyteArray negative, jint width, jint height, jint steps, jdouble cfg,
    jlong seed, jintArray size) {
    brazier::ImageRequest request;
    request.prompt = text(env, prompt);
    request.negative = text(env, negative);
    request.width = width;
    request.height = height;
    request.steps = steps;
    request.cfg = cfg;
    request.seed = seed;

    jmethodID report = env->GetMethodID(env->GetObjectClass(self), "onProgress", "(Ljava/lang/String;JJ)V");
    brazier::Image image;
    std::string error;
    // Generation runs on the calling thread, so its JNIEnv is valid in the callback.
    brazier::Outcome outcome = engine.generate(
        request,
        [&](const std::string& stage, int64_t completed, int64_t total) {
            jstring name = env->NewStringUTF(stage.c_str());
            env->CallVoidMethod(self, report, name, static_cast<jlong>(completed), static_cast<jlong>(total));
            env->DeleteLocalRef(name);
        },
        image, error);
    if (outcome == brazier::Outcome::Cancelled) return nullptr;
    const size_t count = static_cast<size_t>(image.width) * image.height;
    if (outcome == brazier::Outcome::Failed || image.channels != 3 || image.pixels.size() != count * 3) {
        fail(env, "java/lang/RuntimeException", error.empty() ? "Image generation failed." : error);
        return nullptr;
    }
    jintArray pixels = env->NewIntArray(static_cast<jsize>(count));
    if (!pixels) return nullptr;
    jint* out = env->GetIntArrayElements(pixels, nullptr);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t* rgb = &image.pixels[i * 3];
        out[i] = static_cast<jint>(0xFF000000u | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]);
    }
    env->ReleaseIntArrayElements(pixels, out, 0);
    const jint dimensions[2] = {image.width, image.height};
    env->SetIntArrayRegion(size, 0, 2, dimensions);
    return pixels;
}

JNIEXPORT void JNICALL
Java_ai_brazier_sdcpp_BrazierSdcppModule_nativeCancel(JNIEnv*, jobject) {
    engine.cancel();
}

JNIEXPORT void JNICALL
Java_ai_brazier_sdcpp_BrazierSdcppModule_nativeUnload(JNIEnv*, jobject) {
    engine.unload();
}

}  // extern "C"
