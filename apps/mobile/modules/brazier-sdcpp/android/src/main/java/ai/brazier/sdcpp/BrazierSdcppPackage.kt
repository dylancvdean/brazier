package ai.brazier.sdcpp

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

class BrazierSdcppPackage : ReactPackage {
  // The library is built for 64-bit ABIs only; elsewhere the module is absent and the app offers no image models.
  override fun createNativeModules(context: ReactApplicationContext): List<NativeModule> =
    try {
      System.loadLibrary("brazier_sdcpp")
      listOf(BrazierSdcppModule(context))
    } catch (error: UnsatisfiedLinkError) {
      emptyList()
    }

  override fun createViewManagers(context: ReactApplicationContext): List<ViewManager<*, *>> = emptyList()
}
