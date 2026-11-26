package ai.brazier.mobile

import android.content.Context
import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.network.OkHttpClientFactory
import com.facebook.react.modules.network.OkHttpClientProvider
import com.facebook.react.uimanager.ViewManager
import java.net.Socket
import java.security.KeyStore
import java.security.MessageDigest
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.SSLEngine
import javax.net.ssl.SSLSession
import javax.net.ssl.SSLSocket
import javax.net.ssl.TrustManagerFactory
import javax.net.ssl.X509ExtendedTrustManager
import okhttp3.OkHttpClient

// Brazier daemons on a local network serve a self-signed certificate. Pairing
// invites carry its SHA-256, and only that exact certificate is accepted for
// the pinned host and port. Every other host keeps the system's trust policy.
object BrazierTlsPinning {
  private val pins = ConcurrentHashMap<String, String>()

  fun pin(hostAndPort: String, sha256: String) {
    pins[hostAndPort.lowercase(Locale.ROOT)] = sha256.lowercase(Locale.ROOT)
  }

  fun pinFor(host: String?, port: Int): String? =
    host?.let { pins["${it.lowercase(Locale.ROOT)}:$port"] }

  fun install(context: Context) {
    OkHttpClientProvider.setOkHttpClientFactory(object : OkHttpClientFactory {
      override fun createNewNetworkModuleClient(): OkHttpClient {
        val trustManager = PinningTrustManager(systemTrustManager())
        val sslContext = SSLContext.getInstance("TLS")
        sslContext.init(null, arrayOf(trustManager), null)
        return OkHttpClientProvider.createClientBuilder(context)
          .sslSocketFactory(sslContext.socketFactory, trustManager)
          .hostnameVerifier(PinningHostnameVerifier)
          .build()
      }
    })
  }

  private fun systemTrustManager(): X509ExtendedTrustManager {
    val factory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm())
    factory.init(null as KeyStore?)
    return factory.trustManagers.filterIsInstance<X509ExtendedTrustManager>().first()
  }

  private fun sha256(certificate: X509Certificate): String =
    MessageDigest.getInstance("SHA-256").digest(certificate.encoded).joinToString("") { "%02x".format(it) }

  private fun verifyPin(chain: Array<X509Certificate>, pin: String) {
    if (chain.isEmpty() || sha256(chain[0]) != pin) {
      throw CertificateException("The host certificate does not match the pinned Brazier certificate.")
    }
  }

  private class PinningTrustManager(private val system: X509ExtendedTrustManager) : X509ExtendedTrustManager() {
    override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String, socket: Socket) {
      val session = (socket as? SSLSocket)?.handshakeSession
      val pin = pinFor(session?.peerHost, session?.peerPort ?: -1)
      if (pin != null) verifyPin(chain, pin) else system.checkServerTrusted(chain, authType, socket)
    }

    override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String, engine: SSLEngine) {
      val pin = pinFor(engine.peerHost, engine.peerPort)
      if (pin != null) verifyPin(chain, pin) else system.checkServerTrusted(chain, authType, engine)
    }

    override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) =
      system.checkServerTrusted(chain, authType)

    override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String, socket: Socket) =
      system.checkClientTrusted(chain, authType, socket)

    override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String, engine: SSLEngine) =
      system.checkClientTrusted(chain, authType, engine)

    override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) =
      system.checkClientTrusted(chain, authType)

    override fun getAcceptedIssuers(): Array<X509Certificate> = system.acceptedIssuers
  }

  // The trust manager already proved the exact certificate for a pinned host.
  private object PinningHostnameVerifier : HostnameVerifier {
    override fun verify(hostname: String, session: SSLSession): Boolean =
      pinFor(hostname, session.peerPort) != null ||
        HttpsURLConnection.getDefaultHostnameVerifier().verify(hostname, session)
  }
}

class BrazierTlsPinningModule(context: ReactApplicationContext) : ReactContextBaseJavaModule(context) {
  override fun getName() = "BrazierTLSPinning"

  @ReactMethod
  fun pin(hostAndPort: String, sha256: String, promise: Promise) {
    BrazierTlsPinning.pin(hostAndPort, sha256)
    promise.resolve(null)
  }
}

class BrazierTlsPinningPackage : ReactPackage {
  override fun createNativeModules(context: ReactApplicationContext): List<NativeModule> =
    listOf(BrazierTlsPinningModule(context))

  override fun createViewManagers(context: ReactApplicationContext): List<ViewManager<*, *>> = emptyList()
}
