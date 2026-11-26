#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <React/RCTBridgeModule.h>
#import <objc/runtime.h>
#import <os/lock.h>
#import <Security/Security.h>

// Brazier daemons on a local network serve a self-signed certificate. Pairing
// invites carry its SHA-256, and only that exact certificate is accepted for
// the pinned host and port. Every other host keeps the system's trust policy.
static NSMutableDictionary<NSString *, NSString *> *BrazierPins;
static os_unfair_lock BrazierPinsLock = OS_UNFAIR_LOCK_INIT;

static NSString *BrazierPinFor(NSString *host, NSInteger port)
{
  NSString *key = [NSString stringWithFormat:@"%@:%ld", host.lowercaseString, (long)port];
  os_unfair_lock_lock(&BrazierPinsLock);
  NSString *pin = BrazierPins[key];
  os_unfair_lock_unlock(&BrazierPinsLock);
  return pin;
}

static NSString *BrazierLeafSha256(SecTrustRef trust)
{
  CFArrayRef chain = SecTrustCopyCertificateChain(trust);
  if (!chain) return nil;
  NSString *hex = nil;
  if (CFArrayGetCount(chain) > 0) {
    SecCertificateRef leaf = (SecCertificateRef)CFArrayGetValueAtIndex(chain, 0);
    NSData *der = CFBridgingRelease(SecCertificateCopyData(leaf));
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(der.bytes, (CC_LONG)der.length, digest);
    NSMutableString *value = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (int index = 0; index < CC_SHA256_DIGEST_LENGTH; index++) [value appendFormat:@"%02x", digest[index]];
    hex = value;
  }
  CFRelease(chain);
  return hex;
}

typedef void (^BrazierChallengeCompletion)(NSURLSessionAuthChallengeDisposition, NSURLCredential *);

static void BrazierHandleChallenge(id self, SEL _cmd, NSURLSession *session,
                                   NSURLAuthenticationChallenge *challenge, BrazierChallengeCompletion completion)
{
  NSURLProtectionSpace *space = challenge.protectionSpace;
  NSString *pin = [space.authenticationMethod isEqualToString:NSURLAuthenticationMethodServerTrust]
    ? BrazierPinFor(space.host, space.port) : nil;
  if (!pin || !space.serverTrust) {
    completion(NSURLSessionAuthChallengePerformDefaultHandling, nil);
    return;
  }
  if ([BrazierLeafSha256(space.serverTrust) isEqualToString:pin]) {
    completion(NSURLSessionAuthChallengeUseCredential, [NSURLCredential credentialForTrust:space.serverTrust]);
  } else {
    completion(NSURLSessionAuthChallengeCancelAuthenticationChallenge, nil);
  }
}

static void BrazierInstallChallengeHandler(NSString *className)
{
  Class target = NSClassFromString(className);
  SEL selector = @selector(URLSession:didReceiveChallenge:completionHandler:);
  if (!target || class_getInstanceMethod(target, selector)) return;
  class_addMethod(target, selector, (IMP)BrazierHandleChallenge, "v@:@@@?");
}

// expo/fetch and React Native's fetch each own one URLSession delegate. Both
// only forward task-level challenges, so add the session-level server trust
// handler that URLSession consults first.
static void BrazierInstallChallengeHandlers(void)
{
  BrazierInstallChallengeHandler(@"ExpoModulesCore.URLSessionSessionDelegateProxy");
  BrazierInstallChallengeHandler(@"RCTHTTPRequestHandler");
}

// RCT_EXPORT_MODULE defines +load, so install from an image constructor.
__attribute__((constructor)) static void BrazierTLSPinningInitialize(void)
{
  BrazierPins = [NSMutableDictionary new];
  BrazierInstallChallengeHandlers();
}

@interface BrazierTLSPinning : NSObject <RCTBridgeModule>
@end

@implementation BrazierTLSPinning

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

RCT_EXPORT_METHOD(pin:(NSString *)hostAndPort sha256:(NSString *)sha256
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject)
{
  BrazierInstallChallengeHandlers();
  os_unfair_lock_lock(&BrazierPinsLock);
  BrazierPins[hostAndPort.lowercaseString] = sha256.lowercaseString;
  os_unfair_lock_unlock(&BrazierPinsLock);
  resolve(nil);
}

@end
