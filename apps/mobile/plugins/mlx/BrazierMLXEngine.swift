import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

// MLX is built as its own framework because it needs iOS 17, while the app
// supports older versions. The app loads the framework at runtime only where
// it can run and finds this class by name, so nothing here is linked into it.

private struct TokenizerBridge: MLXLMCommon.Tokenizer {
  let upstream: any Tokenizers.Tokenizer

  func encode(text: String, addSpecialTokens: Bool) -> [Int] {
    upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
  }

  func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
    upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
  }

  func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
  func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
  var bosToken: String? { upstream.bosToken }
  var eosToken: String? { upstream.eosToken }
  var unknownToken: String? { upstream.unknownToken }

  func applyChatTemplate(
    messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?
  ) throws -> [Int] {
    do {
      return try upstream.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
    } catch Tokenizers.TokenizerError.missingChatTemplate {
      throw MLXLMCommon.TokenizerError.missingChatTemplate
    }
  }
}

private struct LocalTokenizerLoader: MLXLMCommon.TokenizerLoader {
  func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
    TokenizerBridge(upstream: try await AutoTokenizer.from(modelFolder: directory))
  }
}

/// Runs an MLX model from a directory the app has already downloaded.
@objc(BrazierMLXEngine)
public final class BrazierMLXEngine: NSObject, @unchecked Sendable {
  private let lock = NSLock()
  private var container: ModelContainer?
  private var task: Task<Void, Never>?

  @objc public static func unavailableReason() -> String? {
    #if targetEnvironment(simulator)
    return "MLX needs a real device; the iOS Simulator has no GPU it can use."
    #else
    return nil
    #endif
  }

  @objc public func load(_ directory: String, completion: @escaping @Sendable (String?) -> Void) {
    Task {
      do {
        // Leave room for the app itself: MLX otherwise keeps freed buffers around.
        MLX.GPU.set(cacheLimit: 32 * 1024 * 1024)
        let loaded = try await LLMModelFactory.shared.loadContainer(
          from: URL(fileURLWithPath: directory, isDirectory: true), using: LocalTokenizerLoader())
        lock.withLock { container = loaded }
        completion(nil)
      } catch {
        completion("MLX could not load this model. \(error.localizedDescription)")
      }
    }
  }

  /// `messages` are `role`/`content` pairs. `onText` receives each new piece of the reply; `completion`
  /// receives the whole reply, or what was written before `stop`.
  @objc public func generate(
    _ messages: [[String: String]], onText: @escaping @Sendable (String) -> Void,
    completion: @escaping @Sendable (String?, String?) -> Void
  ) {
    guard let container = lock.withLock({ self.container }) else {
      completion(nil, "The model is not loaded.")
      return
    }
    let chat = messages.compactMap { message -> Chat.Message? in
      guard let content = message["content"] else { return nil }
      switch message["role"] {
      case "system": return .system(content)
      case "user": return .user(content)
      case "assistant": return .assistant(content)
      default: return nil
      }
    }
    let running = Task {
      var reply = ""
      do {
        for try await chunk in ChatSession(container).streamResponse(to: chat) {
          if Task.isCancelled { break }
          reply += chunk
          onText(chunk)
        }
        completion(reply, nil)
      } catch {
        if Task.isCancelled { completion(reply, nil) } else {
          completion(nil, "MLX could not reply. \(error.localizedDescription)")
        }
      }
    }
    lock.withLock { task = running }
  }

  @objc public func stop() {
    lock.withLock { task }?.cancel()
  }

  @objc public func unload() {
    stop()
    lock.withLock { container = nil }
    MLX.GPU.clearCache()
  }
}
