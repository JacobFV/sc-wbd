import { Config } from "@remotion/cli/config";

// Rendered locally on a shared machine: keep concurrency low so the render does
// not compete with training for the unified memory pool. See reports/site.md.
Config.setVideoImageFormat("jpeg");
Config.setConcurrency(2);
Config.setChromiumOpenGlRenderer("swiftshader");
Config.setOverwriteOutput(true);
