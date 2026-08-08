import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { theme } from "./theme";

/** Fade + slight rise. Used for every text element so motion reads as one system. */
export const Rise: React.FC<{
  delay?: number;
  children: React.ReactNode;
  distance?: number;
  style?: React.CSSProperties;
}> = ({ delay = 0, children, distance = 18, style }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({
    frame: frame - delay,
    fps,
    config: { damping: 200, mass: 0.6 },
  });
  return (
    <div
      style={{
        opacity: s,
        transform: `translateY(${(1 - s) * distance}px)`,
        ...style,
      }}
    >
      {children}
    </div>
  );
};

/** A horizontal rule that draws itself. */
export const DrawRule: React.FC<{ delay?: number; width?: number | string; color?: string }> = ({
  delay = 0,
  width = "100%",
  color = theme.ruleStrong,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - delay, fps, config: { damping: 200 } });
  return (
    <div style={{ width, height: 1, background: color, transform: `scaleX(${s})`, transformOrigin: "left" }} />
  );
};

export const Kicker: React.FC<{ children: React.ReactNode; delay?: number }> = ({
  children,
  delay = 0,
}) => (
  <Rise delay={delay}>
    <div
      style={{
        fontFamily: theme.sans,
        fontSize: 26,
        fontWeight: 600,
        letterSpacing: "0.16em",
        textTransform: "uppercase",
        color: theme.ink3,
      }}
    >
      {children}
    </div>
  </Rise>
);

export const Headline: React.FC<{
  children: React.ReactNode;
  delay?: number;
  size?: number;
}> = ({ children, delay = 0, size = 86 }) => (
  <Rise delay={delay}>
    <div
      style={{
        // Sans, matching the site's headings. Body copy stays serif in both
        // places; the split is what makes a heading scannable next to it.
        fontFamily: theme.sans,
        fontSize: size,
        lineHeight: 1.12,
        letterSpacing: "-0.028em",
        color: theme.ink,
        fontWeight: 650,
        maxWidth: 1500,
      }}
    >
      {children}
    </div>
  </Rise>
);

export const Body: React.FC<{
  children: React.ReactNode;
  delay?: number;
  size?: number;
  color?: string;
}> = ({ children, delay = 0, size = 34, color = theme.ink2 }) => (
  <Rise delay={delay}>
    <div
      style={{
        fontFamily: theme.serif,
        fontSize: size,
        lineHeight: 1.5,
        color,
        maxWidth: 1200,
      }}
    >
      {children}
    </div>
  </Rise>
);

/** A number that counts up to its value, with a caption underneath. */
export const CountUp: React.FC<{
  to: number;
  decimals?: number;
  delay?: number;
  duration?: number;
  color?: string;
  size?: number;
  prefix?: string;
  suffix?: string;
}> = ({ to, decimals = 0, delay = 0, duration = 26, color = theme.ink, size = 130, prefix = "", suffix = "" }) => {
  const frame = useCurrentFrame();
  const p = interpolate(frame - delay, [0, duration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // Ease-out so the last digits settle rather than snapping.
  const eased = 1 - Math.pow(1 - p, 3);
  const value = to * eased;
  return (
    <span
      style={{
        fontFamily: theme.mono,
        fontSize: size,
        fontVariantNumeric: "tabular-nums",
        letterSpacing: "-0.03em",
        color,
        lineHeight: 1,
      }}
    >
      {prefix}
      {value.toFixed(decimals)}
      {suffix}
    </span>
  );
};

export const Stat: React.FC<{
  value: React.ReactNode;
  label: React.ReactNode;
  delay?: number;
}> = ({ value, label, delay = 0 }) => (
  <Rise delay={delay} style={{ flex: "0 1 auto" }}>
    <div>{value}</div>
    <div
      style={{
        fontFamily: theme.sans,
        fontSize: 24,
        lineHeight: 1.35,
        color: theme.ink3,
        marginTop: 16,
        maxWidth: 340,
      }}
    >
      {label}
    </div>
  </Rise>
);

/** Monospace source citation, mirroring the site's provenance chip. */
export const Source: React.FC<{ children: React.ReactNode; delay?: number }> = ({
  children,
  delay = 0,
}) => (
  <Rise delay={delay}>
    <div
      style={{
        fontFamily: theme.mono,
        fontSize: 20,
        color: theme.ink3,
        border: `1px solid ${theme.rule}`,
        borderRadius: 4,
        padding: "6px 12px",
        display: "inline-block",
        background: theme.bgSoft,
      }}
    >
      {children}
    </div>
  </Rise>
);

export const Code: React.FC<{ children: React.ReactNode; delay?: number; highlight?: boolean }> = ({
  children,
  delay = 0,
  highlight = false,
}) => (
  <Rise delay={delay}>
    <pre
      style={{
        fontFamily: theme.mono,
        fontSize: 30,
        lineHeight: 1.6,
        color: highlight ? theme.fail : theme.ink,
        background: theme.bgSoft,
        border: `1px solid ${highlight ? theme.fail : theme.rule}`,
        borderRadius: 6,
        padding: "24px 32px",
        margin: 0,
        display: "inline-block",
      }}
    >
      {children}
    </pre>
  </Rise>
);

/** The standard frame: background, generous margin, bottom-left wordmark. */
export const Slide: React.FC<{ children: React.ReactNode; wordmark?: boolean }> = ({
  children,
  wordmark = true,
}) => (
  <AbsoluteFill style={{ background: theme.bg }}>
    <AbsoluteFill
      style={{
        padding: "110px 140px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
      }}
    >
      {children}
    </AbsoluteFill>
    {wordmark ? (
      <div
        style={{
          position: "absolute",
          left: 140,
          bottom: 64,
          fontFamily: theme.sans,
          fontSize: 22,
          fontWeight: 600,
          letterSpacing: "0.14em",
          color: theme.ink3,
        }}
      >
        SC&#8209;WBD
      </div>
    ) : null}
  </AbsoluteFill>
);

/** Cross-fade wrapper so sequences do not hard-cut. */
export const FadeIO: React.FC<{
  children: React.ReactNode;
  durationInFrames: number;
  fade?: number;
}> = ({ children, durationInFrames, fade = 12 }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(
    frame,
    [0, fade, durationInFrames - fade, durationInFrames],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  return <AbsoluteFill style={{ opacity }}>{children}</AbsoluteFill>;
};
