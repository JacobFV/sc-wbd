/**
 * Design tokens shared with the website's stylesheet.
 *
 * These deliberately mirror `site/static/style.css` so a video dropped onto the
 * site does not look like it came from somewhere else. Same restraint: the
 * chrome is monochrome, and colour only ever marks a measured status.
 */

export const theme = {
  bg: "#101215",
  bgSoft: "#171a1e",
  ink: "#e5e7ea",
  ink2: "#a6a9af",
  ink3: "#7e8288",
  rule: "#272b31",
  ruleStrong: "#3b4047",
  pass: "#5fbf8b",
  fail: "#ee7f74",
  unknown: "#90949a",

  serif:
    'Charter, "Bitstream Charter", "Sitka Text", Cambria, Georgia, "Times New Roman", serif',
  sans:
    'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  mono:
    'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
} as const;

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;
