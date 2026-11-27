#import <Foundation/Foundation.h>
#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>
#import <mach/mach.h>
#import <os/proc.h>

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

// MLX lives in an embedded framework that the app does not link, because it
// needs iOS 17. It is loaded on first use, and only where the system can.
static Class BrazierMLXClass(void)
{
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    if (@available(iOS 17.0, *)) {
      NSString *path = [NSBundle.mainBundle.privateFrameworksPath stringByAppendingPathComponent:@"BrazierMLX.framework"];
      [[NSBundle bundleWithPath:path] load];
    }
  });
  return NSClassFromString(@"BrazierMLXEngine");
}

static Class BrazierEngineClass(NSString *name)
{
  if ([name isEqualToString:@"apple"]) return NSClassFromString(@"BrazierAppleModelEngine");
  if ([name isEqualToString:@"mlx"]) return BrazierMLXClass();
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

// What iOS lets this app use before ending it: what it holds now plus what it may still take.
// Null where the system does not say, such as the Simulator.
RCT_EXPORT_METHOD(memoryLimit:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
  size_t available = os_proc_available_memory();
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (available == 0 || task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
    resolve([NSNull null]);
    return;
  }
  resolve(@((double)available + (double)info.phys_footprint));
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
