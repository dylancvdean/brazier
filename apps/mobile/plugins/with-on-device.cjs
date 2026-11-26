const fs = require('node:fs');
const path = require('node:path');
const { IOSConfig, withXcodeProject } = require('expo/config-plugins');

const sources = ['BrazierOnDevice.m', 'BrazierAppleModelEngine.swift'];

// Native engines for "This device" on iOS: the bridge the app talks to and
// Apple's system model. llama.cpp comes from the llama.rn package instead.
module.exports = function withOnDevice(config) {
  return withXcodeProject(config, (config) => {
    const projectName = IOSConfig.XcodeUtils.getProjectName(config.modRequest.projectRoot);
    for (const source of sources) {
      const target = path.join(config.modRequest.platformProjectRoot, projectName, source);
      fs.copyFileSync(path.join(__dirname, 'on-device', source), target);
      if (!config.modResults.hasFile(`${projectName}/${source}`)) {
        IOSConfig.XcodeUtils.addBuildSourceFileToGroup({
          filepath: `${projectName}/${source}`,
          groupName: projectName,
          project: config.modResults,
        });
      }
    }
    return config;
  });
};
