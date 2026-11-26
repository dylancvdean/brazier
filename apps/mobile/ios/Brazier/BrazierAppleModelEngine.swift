import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

/// Apple's on-device system model (Foundation Models). It needs iOS 26 and Apple Intelligence turned on;
/// everywhere else `unavailableReason` says why and the app does not offer it.
@objc(BrazierAppleModelEngine)
final class BrazierAppleModelEngine: NSObject {
  private var task: Task<Void, Never>?

  @objc static func unavailableReason() -> String? {
    #if canImport(FoundationModels)
    guard #available(iOS 26.0, *) else { return "Apple Intelligence needs iOS 26 or later." }
    switch SystemLanguageModel.default.availability {
    case .available: return nil
    case .unavailable(.deviceNotEligible): return "This device does not support Apple Intelligence."
    case .unavailable(.appleIntelligenceNotEnabled): return "Apple Intelligence is turned off in Settings."
    case .unavailable(.modelNotReady): return "The Apple Intelligence model is not ready yet."
    case .unavailable: return "Apple Intelligence is unavailable."
    }
    #else
    return "This build does not include Apple Intelligence."
    #endif
  }

  @objc func load(_ directory: String, completion: @escaping (String?) -> Void) {
    completion(nil)
  }

  /// `messages` are `role`/`content` pairs ending with the user's message. `onText` receives each new piece
  /// of the reply; `completion` receives the whole reply, or what was written before `stop`.
  @objc func generate(
    _ messages: [[String: String]], onText: @escaping (String) -> Void,
    completion: @escaping (String?, String?) -> Void
  ) {
    #if canImport(FoundationModels)
    guard #available(iOS 26.0, *) else {
      completion(nil, "Apple Intelligence needs iOS 26 or later.")
      return
    }
    guard let prompt = messages.last, prompt["role"] == "user", let question = prompt["content"] else {
      completion(nil, "The conversation must end with your message.")
      return
    }
    let text = { (content: String) in [Transcript.Segment.text(.init(content: content))] }
    let entries = messages.dropLast().compactMap { message -> Transcript.Entry? in
      guard let content = message["content"] else { return nil }
      switch message["role"] {
      case "system": return .instructions(.init(segments: text(content), toolDefinitions: []))
      case "user": return .prompt(.init(segments: text(content)))
      case "assistant": return .response(.init(assetIDs: [], segments: text(content)))
      default: return nil
      }
    }
    task = Task {
      var reply = ""
      do {
        let session = LanguageModelSession(transcript: Transcript(entries: entries))
        // Each snapshot is the reply so far, not the new part.
        for try await snapshot in session.streamResponse(to: question) {
          let content = snapshot.content
          if content.hasPrefix(reply) { onText(String(content.dropFirst(reply.count))) }
          reply = content
        }
        completion(reply, nil)
      } catch {
        if Task.isCancelled { completion(reply, nil) } else { completion(nil, Self.describe(error)) }
      }
    }
    #else
    completion(nil, "This build does not include Apple Intelligence.")
    #endif
  }

  #if canImport(FoundationModels)
  @available(iOS 26.0, *)
  private static func describe(_ error: Error) -> String {
    if let failure = error as? LanguageModelSession.GenerationError {
      switch failure {
      case .exceededContextWindowSize: return "This conversation is too long for Apple Intelligence. Start a new chat."
      case .assetsUnavailable: return "The Apple Intelligence model is not downloaded on this device yet."
      case .guardrailViolation: return "Apple Intelligence declined this request."
      default: break
      }
    }
    return "Apple Intelligence could not reply. \(error.localizedDescription)"
  }
  #endif

  @objc func stop() {
    task?.cancel()
  }

  @objc func unload() {
    stop()
  }
}
