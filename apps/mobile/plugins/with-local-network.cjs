const fs = require('node:fs');
const path = require('node:path');
const { IOSConfig, withXcodeProject } = require('expo/config-plugins');

const source = 'BrazierLocalNetwork.m';

// Probes a pairing host before the first request so iOS shows the Local
// Network alert up front and failures can be explained.
module.exports = function withLocalNetwork(config) {
  return withXcodeProject(config, (config) => {
    const projectName = IOSConfig.XcodeUtils.getProjectName(config.modRequest.projectRoot);
    const target = path.join(config.modRequest.platformProjectRoot, projectName, source);
    fs.copyFileSync(path.join(__dirname, 'local-network', source), target);
    if (!config.modResults.hasFile(`${projectName}/${source}`)) {
      IOSConfig.XcodeUtils.addBuildSourceFileToGroup({
        filepath: `${projectName}/${source}`,
        groupName: projectName,
        project: config.modResults,
      });
    }
    return config;
  });
};
