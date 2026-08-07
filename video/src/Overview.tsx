import React from "react";
import { AbsoluteFill, Sequence, interpolate, useCurrentFrame } from "remotion";
import { theme } from "./theme";
import {
  Body,
  CountUp,
  DrawRule,
  FadeIO,
  Headline,
  Kicker,
  Rise,
  Slide,
} from "./components";
import { PARCEL_GROUP, PARCEL_XYZ } from "./parcels";

/**
 * "What SC-WBD is."
 *
 * Product-first: what the system is and what it does. Every number here is
 * traceable to a file in the repository — the 414 = 400 + 14 parcel split from
 * site/static/brain.json and tests/anatomy/test_families.py; the four
 * attachment kinds from scwbd/schema/attachment.py:66; the 5.6% / 51.7% lead
 * field variance pair from reports/anatomy_families.md and
 * scwbd/anatomy/geometry.py:478.
 */

const T1 = 115; // the framing sentence
const T2 = 145; // one brain, 414 parcels
const T3 = 190; // four attachment kinds
const T4 = 125; // EEG alone still trains the shared dynamics
const T5 = 140; // the state carries orientation
const T6 = 145; // what you can do

export const OVERVIEW_DURATION = T1 + T2 + T3 + T4 + T5 + T6;

/* ------------------------------------------------------------------ cloud */

/**
 * The 414 parcel centroids, spun about the superior-inferior axis.
 *
 * Same geometry the website's brain viewer draws. Cortex and subcortex are
 * separated by brightness and size rather than hue, because in this house
 * style colour only ever marks a measured status.
 */
const ParcelCloud: React.FC<{
  size?: number;
  delay?: number;
  spinFrames?: number;
}> = ({ size = 560, delay = 0, spinFrames = 420 }) => {
  const frame = useCurrentFrame();
  const t = frame - delay;

  const theta = ((t / spinFrames) * Math.PI * 2) - 0.6;
  const tilt = 0.34;
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const cosP = Math.cos(tilt);
  const sinP = Math.sin(tilt);

  const reveal = interpolate(t, [0, 46], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const half = size / 2;
  const scale = size * 0.46;
  const focal = 3.2;

  type Dot = { x: number; y: number; r: number; o: number; g: number; d: number };
  const dots: Dot[] = [];

  for (let i = 0; i < PARCEL_GROUP.length; i++) {
    const x = PARCEL_XYZ[i * 3];
    const y = PARCEL_XYZ[i * 3 + 1];
    const z = PARCEL_XYZ[i * 3 + 2];

    // Spin about the superior-inferior axis, then tilt the whole thing forward.
    const xr = x * cosT - y * sinT;
    const yr = x * sinT + y * cosT;
    const yt = yr * cosP - z * sinP;
    const zt = yr * sinP + z * cosP;

    const persp = focal / (focal + yt);
    const g = PARCEL_GROUP[i];

    dots.push({
      x: half + xr * scale * persp,
      y: half - zt * scale * persp,
      r: (g === 1 ? 7.5 : 3.4) * persp,
      o: interpolate(yt, [-0.9, 0.9], [1, 0.34], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp",
      }),
      g,
      d: yt,
    });
  }

  // Painter's algorithm: far parcels first, so the near surface reads as a surface.
  dots.sort((a, b) => b.d - a.d);

  // Parcels fade in from the back of the volume forward.
  const shown = Math.round(dots.length * reveal);

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {dots.slice(0, shown).map((p, i) => (
        <circle
          key={i}
          cx={p.x}
          cy={p.y}
          r={p.r}
          fill={p.g === 1 ? theme.ink : theme.ink2}
          opacity={p.g === 1 ? Math.min(1, p.o + 0.15) : p.o * 0.72}
        />
      ))}
    </svg>
  );
};

/* ------------------------------------------------------------- attachment */

const Attachment: React.FC<{
  name: string;
  gloss: string;
  detail: string;
  delay: number;
}> = ({ name, gloss, detail, delay }) => (
  <Rise delay={delay}>
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: 40,
        borderTop: `1px solid ${theme.rule}`,
        padding: "18px 0",
        width: 1560,
      }}
    >
      <div
        style={{
          fontFamily: theme.mono,
          fontSize: 30,
          color: theme.ink,
          flex: "0 0 330px",
          letterSpacing: "-0.01em",
        }}
      >
        {name}
      </div>
      <div style={{ flex: "1 1 auto" }}>
        <div style={{ fontFamily: theme.serif, fontSize: 34, color: theme.ink, lineHeight: 1.3 }}>
          {gloss}
        </div>
        <div
          style={{
            fontFamily: theme.sans,
            fontSize: 23,
            color: theme.ink3,
            marginTop: 7,
            lineHeight: 1.35,
          }}
        >
          {detail}
        </div>
      </div>
    </div>
  </Rise>
);

/* ------------------------------------------------------------------ bars  */

/** Two measured shares of the whitened EEG lead field, drawn to scale. */
const LeadFieldBars: React.FC<{ delay: number }> = ({ delay }) => {
  const frame = useCurrentFrame();
  const grow = interpolate(frame - delay, [0, 34], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const eased = 1 - Math.pow(1 - grow, 3);

  const rows = [
    { label: "per-parcel scalar", value: 5.6, color: theme.ink3 },
    { label: "3-vector current-dipole moment", value: 51.7, color: theme.ink },
  ];
  const track = 1180;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 34 }}>
      {rows.map((r) => (
        <div key={r.label}>
          <div
            style={{
              display: "flex",
              alignItems: "baseline",
              justifyContent: "space-between",
              width: track,
              marginBottom: 12,
            }}
          >
            <span style={{ fontFamily: theme.sans, fontSize: 26, color: theme.ink2 }}>
              {r.label}
            </span>
            <span
              style={{
                fontFamily: theme.mono,
                fontSize: 34,
                color: r.color,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {(r.value * eased).toFixed(1)}%
            </span>
          </div>
          <div style={{ width: track, height: 16, background: theme.bgSoft, borderRadius: 2 }}>
            <div
              style={{
                width: (track * r.value * eased) / 60,
                height: 16,
                background: r.color,
                borderRadius: 2,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
};

/* ------------------------------------------------------------ capability  */

const Capability: React.FC<{ children: React.ReactNode; delay: number }> = ({
  children,
  delay,
}) => (
  <Rise delay={delay}>
    <div
      style={{
        display: "flex",
        gap: 26,
        alignItems: "baseline",
        padding: "13px 0",
        maxWidth: 1520,
      }}
    >
      <div style={{ fontFamily: theme.mono, fontSize: 30, color: theme.ink3 }}>&rarr;</div>
      <div style={{ fontFamily: theme.serif, fontSize: 38, color: theme.ink, lineHeight: 1.32 }}>
        {children}
      </div>
    </div>
  </Rise>
);

/* ----------------------------------------------------------------- video  */

export const Overview: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: theme.bg }}>
      <Sequence durationInFrames={T1}>
        <FadeIO durationInFrames={T1}>
          <Slide wordmark={false}>
            <Rise delay={0}>
              <div
                style={{
                  fontFamily: theme.sans,
                  fontSize: 34,
                  fontWeight: 600,
                  letterSpacing: "0.22em",
                  color: theme.ink3,
                }}
              >
                SC&#8209;WBD
              </div>
            </Rise>
            <div style={{ height: 40 }} />
            <Headline delay={8} size={80}>
              An integrated whole-brain foundation model across modalities, scales
              and dynamics.
            </Headline>
            <div style={{ height: 52 }} />
            <div style={{ width: 900 }}>
              <DrawRule delay={34} />
            </div>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={T1} durationInFrames={T2}>
        <FadeIO durationInFrames={T2}>
          <Slide>
            <div style={{ display: "flex", alignItems: "center", gap: 100 }}>
              <div style={{ flex: "1 1 auto" }}>
                <Kicker delay={0}>One brain, one state</Kicker>
                <div style={{ height: 40 }} />
                <Headline delay={8} size={68}>
                  One brain, modelled as a 414&#8209;parcel dynamical system.
                </Headline>
                <div style={{ height: 46 }} />
                <Rise delay={30}>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 22 }}>
                    <CountUp to={400} delay={34} duration={30} color={theme.ink} size={82} />
                    <span
                      style={{ fontFamily: theme.sans, fontSize: 27, color: theme.ink3 }}
                    >
                      cortical
                    </span>
                    <span
                      style={{ fontFamily: theme.mono, fontSize: 54, color: theme.ink3 }}
                    >
                      +
                    </span>
                    <CountUp to={14} delay={34} duration={30} color={theme.ink} size={82} />
                    <span
                      style={{ fontFamily: theme.sans, fontSize: 27, color: theme.ink3 }}
                    >
                      subcortical
                    </span>
                  </div>
                </Rise>
              </div>
              <Rise delay={14} distance={0}>
                <ParcelCloud size={600} delay={14} />
              </Rise>
            </div>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={T1 + T2} durationInFrames={T3}>
        <FadeIO durationInFrames={T3}>
          <Slide>
            <Kicker delay={0}>Every modality supervises the part it observes</Kicker>
            <div style={{ height: 30 }} />
            <Headline delay={6} size={58}>
              Four attachment kinds.
            </Headline>
            <div style={{ height: 34 }} />
            <Attachment
              delay={22}
              name="stimulus"
              gloss="The world driving the subject."
              detail="Audio, video, text, task events."
            />
            <Attachment
              delay={40}
              name="observation"
              gloss="A measurement of the brain through a declared forward operator."
              detail="EEG via a lead field, BOLD via a haemodynamic model, MEG."
            />
            <Attachment
              delay={58}
              name="boundary_output"
              gloss="Produced by the subject, measured outside the skull."
              detail="Eye tracking, motor, speech, ECG, cognitive-test responses."
            />
            <Attachment
              delay={76}
              name="context"
              gloss="Slow conditioning."
              detail="Time of day, session, drug state."
            />
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={T1 + T2 + T3} durationInFrames={T4}>
        <FadeIO durationInFrames={T4}>
          <Slide>
            <Kicker delay={0}>Partial coverage is the normal case</Kicker>
            <div style={{ height: 42 }} />
            <Headline delay={8} size={66}>
              A subject measured with EEG alone still trains the shared dynamics.
            </Headline>
            <div style={{ height: 50 }} />
            <div style={{ width: 800 }}>
              <DrawRule delay={34} />
            </div>
            <div style={{ height: 38 }} />
            <Body delay={40} size={38}>
              Missing states are marginalised, never imputed.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={T1 + T2 + T3 + T4} durationInFrames={T5}>
        <FadeIO durationInFrames={T5}>
          <Slide>
            <Kicker delay={0}>The state carries orientation</Kicker>
            <div style={{ height: 34 }} />
            <Headline delay={6} size={54}>
              Share of the whitened EEG lead field a parcel can express.
            </Headline>
            <div style={{ height: 48 }} />
            <LeadFieldBars delay={22} />
            <div style={{ height: 46 }} />
            <Body delay={62} size={36} color={theme.ink}>
              Orientation buys about 9&#215; what spatial resolution buys.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={T1 + T2 + T3 + T4 + T5} durationInFrames={T6}>
        <FadeIO durationInFrames={T6}>
          <Slide wordmark={false}>
            <Kicker delay={0}>What you can do with it</Kicker>
            <div style={{ height: 40 }} />
            <Capability delay={10}>
              Roll the state forward and predict across modalities.
            </Capability>
            <Capability delay={26}>
              Put a subject&#8217;s fMRI into the 414-parcel space.
            </Capability>
            <Capability delay={42}>
              Build a lead field on their own head geometry.
            </Capability>
            <Capability delay={58}>
              Compare TMS and tFUS targets under an E-field model.
            </Capability>
            <div style={{ height: 54 }} />
            <div style={{ width: 700 }}>
              <DrawRule delay={78} />
            </div>
            <div style={{ height: 32 }} />
            <Rise delay={86}>
              <div
                style={{
                  fontFamily: theme.sans,
                  fontSize: 30,
                  letterSpacing: "0.05em",
                  color: theme.ink2,
                }}
              >
                SC-WBD &mdash; modalities, scales and dynamics in one model.
              </div>
            </Rise>
          </Slide>
        </FadeIO>
      </Sequence>
    </AbsoluteFill>
  );
};
