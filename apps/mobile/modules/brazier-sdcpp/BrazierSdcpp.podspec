Pod::Spec.new do |s|
  s.name = 'BrazierSdcpp'
  s.version = '0.1.0'
  s.summary = 'On-device image generation with stable-diffusion.cpp.'
  s.homepage = 'https://github.com/dylancvdean/brazier'
  s.license = 'MIT'
  s.author = 'Brazier'
  s.source = { :path => '.' }
  s.platform = :ios, '17.0'
  s.source_files = 'ios/*.{h,mm}', 'cpp/*.{h,cpp}'
  s.private_header_files = 'cpp/*.h'
  # Built by scripts/build-ios.sh, which the Podfile runs before installing.
  s.vendored_frameworks = 'build/ios/sdcpp.xcframework'
  s.frameworks = 'Accelerate', 'Foundation', 'Metal', 'MetalKit', 'UIKit'
  s.libraries = 'c++'
  # The library is built for arm64 only, so Intel simulators are left out.
  arm64_only = { 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'x86_64' }
  s.pod_target_xcconfig = { 'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17' }.merge(arm64_only)
  s.user_target_xcconfig = arm64_only
  s.dependency 'React-Core'
end
