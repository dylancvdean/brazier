require 'xcodeproj'

# Adds the BrazierMLX framework target to the app's Xcode project. The Podfile
# calls this after every `pod install`; it does nothing once the target exists.
#
# MLX needs iOS 17 and the app supports 15.1, so the app cannot link it. The
# framework is built with its own deployment target and only embedded; the app
# loads it at runtime where it can run (see BrazierOnDevice.m).
module BrazierMLX
  NAME = 'BrazierMLX'.freeze
  PACKAGES = [
    ['https://github.com/ml-explore/mlx-swift-lm', '3.33.0', %w[MLXLLM MLXLMCommon]],
    ['https://github.com/ml-explore/mlx-swift', '0.33.0', %w[MLX]],
    ['https://github.com/huggingface/swift-transformers', '1.4.0', %w[Tokenizers]],
  ].freeze

  def self.ensure_target(project_path)
    project = Xcodeproj::Project.open(project_path)
    return if project.targets.any? { |target| target.name == NAME }

    app = project.targets.find { |target| target.name == 'Brazier' }
    team = app.build_configurations.first.build_settings['DEVELOPMENT_TEAM']
    target = project.new_target(:framework, NAME, :ios, '17.0', nil, :swift)
    target.build_configurations.each do |config|
      config.build_settings.merge!(
        'PRODUCT_BUNDLE_IDENTIFIER' => 'ai.brazier.mobile.mlx',
        'PRODUCT_NAME' => NAME,
        'GENERATE_INFOPLIST_FILE' => 'YES',
        'MARKETING_VERSION' => '1.0',
        'CURRENT_PROJECT_VERSION' => '1',
        'SWIFT_VERSION' => '5.0',
        'IPHONEOS_DEPLOYMENT_TARGET' => '17.0',
        'TARGETED_DEVICE_FAMILY' => '1,2',
        'SKIP_INSTALL' => 'YES',
        'DEFINES_MODULE' => 'YES',
        'CODE_SIGN_STYLE' => 'Automatic',
        # MLX is for Apple silicon only; the Intel simulator slice is not built.
        'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'x86_64',
        'ENABLE_USER_SCRIPT_SANDBOXING' => 'NO',
      )
      config.build_settings['DEVELOPMENT_TEAM'] = team if team
    end

    group = project.main_group.new_group(NAME, '../plugins/mlx')
    target.add_file_references([group.new_file('BrazierMLXEngine.swift')])

    PACKAGES.each do |url, version, products|
      package = project.new(Xcodeproj::Project::Object::XCRemoteSwiftPackageReference)
      package.repositoryURL = url
      package.requirement = { 'kind' => 'exactVersion', 'version' => version }
      project.root_object.package_references << package
      products.each do |name|
        product = project.new(Xcodeproj::Project::Object::XCSwiftPackageProductDependency)
        product.package = package
        product.product_name = name
        target.package_product_dependencies << product
        file = project.new(Xcodeproj::Project::Object::PBXBuildFile)
        file.product_ref = product
        target.frameworks_build_phase.files << file
      end
    end

    # Xcode copies a package's resources into apps only. MLX looks for its Metal library next to the
    # binary, and the tokenizer package looks for its bundle in the framework that contains its code.
    resources = target.new_shell_script_build_phase('Copy package resources')
    resources.shell_script = <<~SCRIPT
      set -e
      cp "${BUILT_PRODUCTS_DIR}/mlx-swift_Cmlx.bundle/default.metallib" "${TARGET_BUILD_DIR}/${WRAPPER_NAME}/mlx.metallib"
      rsync -a --delete "${BUILT_PRODUCTS_DIR}/swift-transformers_Hub.bundle" "${TARGET_BUILD_DIR}/${WRAPPER_NAME}/"
    SCRIPT
    resources.input_paths = [
      '$(BUILT_PRODUCTS_DIR)/mlx-swift_Cmlx.bundle/default.metallib',
      '$(BUILT_PRODUCTS_DIR)/swift-transformers_Hub.bundle',
    ]
    resources.output_paths = [
      '$(TARGET_BUILD_DIR)/$(WRAPPER_NAME)/mlx.metallib',
      '$(TARGET_BUILD_DIR)/$(WRAPPER_NAME)/swift-transformers_Hub.bundle',
    ]

    # Embedded, not linked: the app must still start on iOS versions that cannot load it.
    app.add_dependency(target)
    embed = app.new_copy_files_build_phase('Embed MLX')
    embed.symbol_dst_subfolder_spec = :frameworks
    embedded = embed.add_file_reference(target.product_reference)
    embedded.settings = { 'ATTRIBUTES' => %w[CodeSignOnCopy RemoveHeadersOnCopy] }

    project.save
  end
end
