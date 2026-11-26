#import <Foundation/Foundation.h>
#import <Network/Network.h>
#import <React/RCTBridgeModule.h>
#import <arpa/inet.h>
#import <ifaddrs.h>
#import <net/if.h>

// iOS shows the Local Network alert the first time the app connects to a host
// on its Wi-Fi or Ethernet subnet, and only then: hosts reached through a
// router, a VPN, or cellular never prompt. A Network framework connection
// triggers the alert, keeps waiting while it is shown, and reports the path it
// took, so the app can ask for access or explain why the host is unreachable.

static NSString *BrazierInterfaceType(nw_path_t path)
{
  if (!path) return @"none";
  if (nw_path_uses_interface_type(path, nw_interface_type_wifi)) return @"wifi";
  if (nw_path_uses_interface_type(path, nw_interface_type_wired)) return @"wired";
  if (nw_path_uses_interface_type(path, nw_interface_type_cellular)) return @"cellular";
  if (nw_path_uses_interface_type(path, nw_interface_type_loopback)) return @"loopback";
  if (nw_path_uses_interface_type(path, nw_interface_type_other)) return @"other";
  return @"none";
}

/** IPv4 subnets of the active broadcast-capable (Wi-Fi or Ethernet) interfaces. */
static NSArray<NSDictionary *> *BrazierLocalNetworks(void)
{
  NSMutableArray<NSDictionary *> *networks = [NSMutableArray new];
  struct ifaddrs *list = NULL;
  if (getifaddrs(&list) != 0) return networks;
  for (struct ifaddrs *entry = list; entry; entry = entry->ifa_next) {
    if (!entry->ifa_addr || !entry->ifa_netmask || entry->ifa_addr->sa_family != AF_INET) continue;
    unsigned int flags = entry->ifa_flags;
    if (!(flags & IFF_UP) || !(flags & IFF_RUNNING) || !(flags & IFF_BROADCAST) || (flags & IFF_LOOPBACK)) continue;
    char address[INET_ADDRSTRLEN];
    char netmask[INET_ADDRSTRLEN];
    if (!inet_ntop(AF_INET, &((struct sockaddr_in *)entry->ifa_addr)->sin_addr, address, sizeof address) ||
        !inet_ntop(AF_INET, &((struct sockaddr_in *)entry->ifa_netmask)->sin_addr, netmask, sizeof netmask)) continue;
    [networks addObject:@{@"address": @(address), @"netmask": @(netmask)}];
  }
  freeifaddrs(list);
  return networks;
}

static BOOL BrazierRefused(nw_error_t error)
{
  return error && nw_error_get_error_domain(error) == nw_error_domain_posix && nw_error_get_error_code(error) == ECONNREFUSED;
}

static id BrazierErrorText(nw_error_t error)
{
  if (!error) return [NSNull null];
  NSError *value = CFBridgingRelease(nw_error_copy_cf_error(error));
  return value.localizedDescription ?: [NSNull null];
}

@interface BrazierLocalNetwork : NSObject <RCTBridgeModule>
@end

@implementation BrazierLocalNetwork

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

RCT_EXPORT_METHOD(probe:(NSString *)host port:(nonnull NSNumber *)port timeoutMs:(nonnull NSNumber *)timeoutMs
                  resolve:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject)
{
  nw_endpoint_t endpoint = nw_endpoint_create_host(host.UTF8String, port.stringValue.UTF8String);
  nw_parameters_t parameters = nw_parameters_create_secure_tcp(NW_PARAMETERS_DISABLE_PROTOCOL,
                                                               NW_PARAMETERS_DEFAULT_CONFIGURATION);
  nw_connection_t connection = nw_connection_create(endpoint, parameters);
  dispatch_queue_t queue = dispatch_queue_create("ai.brazier.local-network-probe", DISPATCH_QUEUE_SERIAL);
  __block BOOL finished = NO;
  __block BOOL denied = NO;
  __block NSString *interface = @"none";
  __block id lastError = [NSNull null];
  void (^finish)(NSString *) = ^(NSString *status) {
    if (finished) return;
    finished = YES;
    nw_connection_cancel(connection);
    resolve(@{@"status": status, @"interface": interface, @"error": lastError, @"networks": BrazierLocalNetworks()});
  };
  nw_connection_set_queue(connection, queue);
  nw_connection_set_state_changed_handler(connection, ^(nw_connection_state_t state, nw_error_t error) {
    nw_path_t path = nw_connection_copy_current_path(connection);
    if (path) interface = BrazierInterfaceType(path);
    if (error) lastError = BrazierErrorText(error);
    switch (state) {
      case nw_connection_state_ready:
        finish(@"ready");
        break;
      case nw_connection_state_waiting:
        // While the alert is up, or after the user denied it, the connection
        // waits with this reason and retries by itself once access is granted.
        if (path && nw_path_get_status(path) == nw_path_status_unsatisfied &&
            nw_path_get_unsatisfied_reason(path) == nw_path_unsatisfied_reason_local_network_denied) {
          denied = YES;
        } else if (BrazierRefused(error)) {
          finish(@"refused");
        }
        break;
      case nw_connection_state_failed:
        finish(BrazierRefused(error) ? @"refused" : @"failed");
        break;
      case nw_connection_state_cancelled:
        nw_connection_set_state_changed_handler(connection, NULL);
        break;
      default:
        break;
    }
  });
  nw_connection_start(connection);
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeoutMs.doubleValue * NSEC_PER_MSEC)), queue, ^{
    finish(denied ? @"denied" : @"timeout");
  });
}

@end
