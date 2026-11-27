#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>
#import <CommonCrypto/CommonDigest.h>
#import <UIKit/UIKit.h>

#include "brazier_sd.h"

static NSString *const BrazierSdcppProgress = @"BrazierSdcppProgress";

// ggml's Metal backend needs GPU features that arrived with the A14. On older
// chips, and in the Simulator, unsupported operations fall back to the CPU in a
// way that crashes, so those run on the CPU entirely.
static BOOL BrazierSdcppUsesMetal(void)
{
#if TARGET_OS_SIMULATOR
  return NO;
#else
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device != nil && [device supportsFamily:MTLGPUFamilyApple7];
#endif
}

@interface BrazierSdcpp : RCTEventEmitter <RCTBridgeModule>
@end

@implementation BrazierSdcpp {
  brazier::ImageEngine _engine;
  dispatch_queue_t _queue;
  BOOL _listening;
}

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

- (instancetype)init
{
  if ((self = [super init])) {
    _queue = dispatch_queue_create("ai.brazier.sdcpp", DISPATCH_QUEUE_SERIAL);
  }
  return self;
}

- (NSArray<NSString *> *)supportedEvents
{
  return @[BrazierSdcppProgress];
}

- (void)startObserving
{
  _listening = YES;
}

- (void)stopObserving
{
  _listening = NO;
}

RCT_EXPORT_METHOD(load:(NSString *)path
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
  dispatch_async(_queue, ^{
    std::string error;
    // Leave the efficiency cores to the rest of the app.
    int threads = MAX(1, (int)NSProcessInfo.processInfo.activeProcessorCount - 2);
    if (self->_engine.load(path.UTF8String, BrazierSdcppUsesMetal(), threads, error)) {
      resolve(nil);
    } else {
      reject(@"load", @(error.c_str()), nil);
    }
  });
}

RCT_EXPORT_METHOD(generate:(NSDictionary *)options
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
  dispatch_async(_queue, ^{
    brazier::ImageRequest request;
    request.prompt = [options[@"prompt"] UTF8String] ?: "";
    request.negative = [options[@"negative"] UTF8String] ?: "";
    request.width = [options[@"width"] intValue];
    request.height = [options[@"height"] intValue];
    request.steps = [options[@"steps"] intValue];
    request.cfg = options[@"cfg"] ? [options[@"cfg"] doubleValue] : -1;
    request.seed = options[@"seed"] ? [options[@"seed"] longLongValue] : -1;
    NSString *output = options[@"output"];
    if ([output hasPrefix:@"file://"]) output = [NSURL URLWithString:output].path;

    brazier::Image image;
    std::string error;
    __weak BrazierSdcpp *weak = self;
    brazier::Outcome outcome = self->_engine.generate(
      request,
      [weak](const std::string &stage, int64_t completed, int64_t total) {
        BrazierSdcpp *module = weak;
        if (module && module->_listening) {
          [module sendEventWithName:BrazierSdcppProgress
                               body:@{@"stage": @(stage.c_str()), @"completed": @(completed), @"total": @(total)}];
        }
      },
      image, error);
    if (outcome == brazier::Outcome::Cancelled) {
      reject(@"cancelled", @"Generation cancelled.", nil);
      return;
    }
    if (outcome == brazier::Outcome::Failed) {
      reject(@"generate", @(error.c_str()), nil);
      return;
    }
    NSData *png = BrazierSdcppPNG(image);
    if (!png || ![png writeToFile:output atomically:YES]) {
      reject(@"generate", @"Could not save the generated image.", nil);
      return;
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(png.bytes, (CC_LONG)png.length, digest);
    NSMutableString *sha256 = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) [sha256 appendFormat:@"%02x", digest[i]];
    resolve(@{@"width": @(image.width), @"height": @(image.height), @"size": @(png.length), @"sha256": sha256});
  });
}

RCT_EXPORT_METHOD(cancel)
{
  _engine.cancel();
}

RCT_EXPORT_METHOD(unload:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
  dispatch_async(_queue, ^{
    self->_engine.unload();
    resolve(nil);
  });
}

static NSData *BrazierSdcppPNG(const brazier::Image &image)
{
  if (image.channels != 3 || image.width <= 0 || image.height <= 0 ||
      image.pixels.size() != (size_t)image.width * image.height * 3) {
    return nil;
  }
  CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, image.pixels.data(), image.pixels.size(), NULL);
  CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
  CGImageRef pixels = CGImageCreate(image.width, image.height, 8, 24, (size_t)image.width * 3, space,
                                    (CGBitmapInfo)kCGImageAlphaNone, provider, NULL, false, kCGRenderingIntentDefault);
  NSData *png = pixels ? UIImagePNGRepresentation([UIImage imageWithCGImage:pixels]) : nil;
  CGImageRelease(pixels);
  CGColorSpaceRelease(space);
  CGDataProviderRelease(provider);
  return png;
}

@end
