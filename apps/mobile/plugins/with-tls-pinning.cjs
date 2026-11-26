const fs = require('node:fs');
const path = require('node:path');
const { IOSConfig, withDangerousMod, withMainApplication, withXcodeProject } = require('expo/config-plugins');

const sources = path.join(__dirname, 'tls-pinning');
const iosSource = 'BrazierTLSPinning.m';
const androidSource = 'BrazierTlsPinning.kt';

// Pairing invites pin the daemon's self-signed certificate. These native
// sources make expo/fetch accept exactly that certificate for its host.
module.exports = function withTlsPinning(config) {
  config = withXcodeProject(config, (config) => {
    const projectName = IOSConfig.XcodeUtils.getProjectName(config.modRequest.projectRoot);
    const target = path.join(config.modRequest.platformProjectRoot, projectName, iosSource);
    fs.copyFileSync(path.join(sources, iosSource), target);
    if (!config.modResults.hasFile(`${projectName}/${iosSource}`)) {
      IOSConfig.XcodeUtils.addBuildSourceFileToGroup({
        filepath: `${projectName}/${iosSource}`,
        groupName: projectName,
        project: config.modResults,
      });
    }
    return config;
  });
  config = withDangerousMod(config, ['android', (config) => {
    const directory = path.join(config.modRequest.platformProjectRoot, 'app/src/main/java/ai/brazier/mobile');
    fs.mkdirSync(directory, { recursive: true });
    fs.copyFileSync(path.join(sources, androidSource), path.join(directory, androidSource));
    return config;
  }]);
  return withMainApplication(config, (config) => {
    if (config.modResults.language !== 'kt') {
      throw new Error('The TLS pinning plugin requires a Kotlin MainApplication.');
    }
    let source = config.modResults.contents;
    const packages = 'PackageList(this).packages.apply {';
    const created = 'super.onCreate()';
    if (!source.includes(packages) || !source.includes(created)) {
      throw new Error('Expo MainApplication template changed; review the TLS pinning registration.');
    }
    if (!source.includes('BrazierTlsPinningPackage()')) {
      source = source.replace(packages, `${packages}\n              add(BrazierTlsPinningPackage())`);
    }
    if (!source.includes('BrazierTlsPinning.install(this)')) {
      source = source.replace(created, `${created}\n    BrazierTlsPinning.install(this)`);
    }
    config.modResults.contents = source;
    return config;
  });
};
