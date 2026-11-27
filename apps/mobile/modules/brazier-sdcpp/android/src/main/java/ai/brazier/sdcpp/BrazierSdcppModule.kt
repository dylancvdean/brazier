package ai.brazier.sdcpp

import android.graphics.Bitmap
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.modules.core.DeviceEventManagerModule
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.Executors

/** On-device image generation with stable-diffusion.cpp. One model, one job at a time. */
class BrazierSdcppModule(private val context: ReactApplicationContext) : ReactContextBaseJavaModule(context) {
  private val worker = Executors.newSingleThreadExecutor()

  override fun getName() = "BrazierSdcpp"

  @ReactMethod
  fun load(path: String, promise: Promise) {
    worker.execute {
      try {
        // Leave two cores to the rest of the app.
        nativeLoad(path.removePrefix("file://"), maxOf(1, Runtime.getRuntime().availableProcessors() - 2))
        promise.resolve(null)
      } catch (error: Throwable) {
        promise.reject("load", error.message ?: "Could not open the model.", error)
      }
    }
  }

  @ReactMethod
  fun generate(options: ReadableMap, promise: Promise) {
    worker.execute {
      try {
        val size = IntArray(2)
        val pixels = nativeGenerate(
          options.getString("prompt") ?: "",
          options.getString("negative") ?: "",
          options.getInt("width"),
          options.getInt("height"),
          if (options.hasKey("steps")) options.getInt("steps") else 0,
          if (options.hasKey("cfg")) options.getDouble("cfg") else -1.0,
          if (options.hasKey("seed")) options.getDouble("seed").toLong() else -1L,
          size,
        )
        if (pixels == null) {
          promise.reject("cancelled", "Generation cancelled.")
          return@execute
        }
        val file = File(options.getString("output")!!.removePrefix("file://"))
        val bitmap = Bitmap.createBitmap(pixels, size[0], size[1], Bitmap.Config.ARGB_8888)
        file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        bitmap.recycle()
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
          val buffer = ByteArray(1 shl 16)
          while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            digest.update(buffer, 0, read)
          }
        }
        promise.resolve(Arguments.createMap().apply {
          putInt("width", size[0])
          putInt("height", size[1])
          putDouble("size", file.length().toDouble())
          putString("sha256", digest.digest().joinToString("") { "%02x".format(it) })
        })
      } catch (error: Throwable) {
        promise.reject("generate", error.message ?: "Image generation failed.", error)
      }
    }
  }

  @ReactMethod
  fun cancel() = nativeCancel()

  @ReactMethod
  fun unload(promise: Promise) {
    worker.execute {
      nativeUnload()
      promise.resolve(null)
    }
  }

  @ReactMethod
  fun addListener(eventName: String) = Unit

  @ReactMethod
  fun removeListeners(count: Int) = Unit

  @Suppress("unused") // Called from jni.cpp.
  private fun onProgress(stage: String, completed: Long, total: Long) {
    context.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java).emit(
      "BrazierSdcppProgress",
      Arguments.createMap().apply {
        putString("stage", stage)
        putDouble("completed", completed.toDouble())
        putDouble("total", total.toDouble())
      },
    )
  }

  private external fun nativeLoad(path: String, threads: Int)
  private external fun nativeGenerate(
    prompt: String, negative: String, width: Int, height: Int, steps: Int, cfg: Double, seed: Long, size: IntArray,
  ): IntArray?
  private external fun nativeCancel()
  private external fun nativeUnload()

  companion object {
    init {
      System.loadLibrary("brazier_sdcpp")
    }
  }
}
