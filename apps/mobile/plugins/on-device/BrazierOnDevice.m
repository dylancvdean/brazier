#import <Foundation/Foundation.h>
#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

// What a native on-device engine implements. The engines are looked up by class
// name so that one that this iOS version cannot load is simply absent.
@protocol BrazierOnDeviceEngine <NSObject>
+ (nullable NSString *)unavailableReason;
- (void)load:(NSString *)directory completion:(void (^)(NSString *_Nullable error))completion;
- (void)generate:(NSArray<NSDictionary<NSString *, NSString *> *> *)messages
          onText:(void (^)(NSString *text))onText
      completion:(void (^)(NSString *_Nullable reply, NSString *_Nullable error))completion;
- (void)stop;
- (void)unload;
@end

static NSString *const BrazierOnDeviceText = @"BrazierOnDeviceText";

@interface BrazierOnDevice : RCTEventEmitter <RCTBridgeModule>
@end

@implementation BrazierOnDevice {
  NSMutableDictionary<NSString *, id<BrazierOnDeviceEngine>> *_engines;
  BOOL _listening;
}

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

- (NSArray<NSString *> *)supportedEvents
{
  return @[BrazierOnDeviceText];
}

- (void)startObserving
{
  _listening = YES;
}

- (void)stopObserving
{
  _listening = NO;
}

static Class BrazierEngineClass(NSString *name)
{
  if ([name isEqualToString:@"apple"]) return NSClassFromString(@"BrazierAppleModelEngine");
  return Nil;
}

- (nullable id<BrazierOnDeviceEngine>)engine:(NSString *)name
{
  @synchronized (self) {
    if (!_engines) _engines = [NSMutableDictionary new];
    id<BrazierOnDeviceEngine> engine = _engines[name];
    if (engine) return engine;
    Class type = BrazierEngineClass(name);
    if (!type || [type unavailableReason]) return nil;
    engine = [type new];
    _engines[name] = engine;
    return engine;
  }
}

RCT_EXPORT_METHOD(availability:(NSString *)name
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
  Class type = BrazierEngineClass(name);
  NSString *reason = type ? [type unavailableReason] : @"This build does not include this engine.";
  resolve(@{ @"available": @(reason == nil), @"reason": reason ?: @"" });
}

RCT_EXPORT_METHOD(load:(NSString *)name
                  directory:(NSString *)directory
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
  id<BrazierOnDeviceEngine> engine = [self engine:name];
  if (!engine) {
    reject(@"unavailable", @"This engine cannot run on this device.", nil);
    return;
  }
  [engine load:directory completion:^(NSString *error) {
    if (error) reject(@"load_failed", error, nil);
    else resolve(nil);
  }];
}

RCT_EXPORT_METHOD(generate:(NSString *)name
                  request:(NSString *)request
                  messages:(NSArray *)messages
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
  id<BrazierOnDeviceEngine> engine = [self engine:name];
  if (!engine) {
    reject(@"unavailable", @"This engine cannot run on this device.", nil);
    return;
  }
  __weak BrazierOnDevice *weakSelf = self;
  [engine generate:messages onText:^(NSString *text) {
    BrazierOnDevice *module = weakSelf;
    if (module && module->_listening && text.length) {
      [module sendEventWithName:BrazierOnDeviceText body:@{ @"request": request, @"text": text }];
    }
  } completion:^(NSString *reply, NSString *error) {
    if (error) reject(@"generate_failed", error, nil);
    else resolve(reply ?: @"");
  }];
}

RCT_EXPORT_METHOD(stop:(NSString *)name)
{
  [[self engine:name] stop];
}

RCT_EXPORT_METHOD(unload:(NSString *)name
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
  @synchronized (self) {
    [_engines[name] unload];
    [_engines removeObjectForKey:name];
  }
  resolve(nil);
}

@end
