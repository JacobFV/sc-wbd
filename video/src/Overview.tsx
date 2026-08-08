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
 * Written for someone who has never read a neuroscience paper.
 *
 * The previous cut opened on "an integrated whole-brain foundation model across
 * modalities, scales and dynamics" and spent a third of its running time on the
 * schema's four attachment kinds and the whitened lead field. Both are true and
 * neither answers the only question a first-time viewer is actually asking,
 * which is what this is for. The arc is now problem -> model -> mechanism ->
 * use -> invitation, and the jargon is spent only where a plain word would be
 * less honest rather than merely less impressive.
 *
 * The numbers are still traceable: 414 = 400 + 14 from site/static/brain.json
 * and tests/anatomy/test_families.py; the sampling rates from the corpus
 * manifests; the closing status from reports/run2_eval.md.
 */

const T1 = 165; // the brain is complicated and no instrument sees it whole
const T2 = 250; // three instruments, three disagreements
const T3 = 150; // so results do not add up, and almost none of it is about you
const T4 = 165; // the model
const T5 = 190; // how it works: 414 regions
const T6 = 215; // how it works: every signal constrains what it can speak to
const T7 = 235; // what you would use it for
const T8 = 190; // the honest status, and the invitation

export const OVERVIEW_DURATION = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8;

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

/* ------------------------------------------------------------- instrument */

/**
 * One way of looking at a brain, with the two things about it that make the
 * others hard to combine with: how fast it reads, and how much it can see.
 * The point of the row is the mismatch down the columns, so the columns are
 * aligned and fixed-width rather than sized to their contents.
 */
const Instrument: React.FC<{
  name: string;
  what: string;
  speed: string;
  reach: string;
  delay: number;
}> = ({ name, what, speed, reach, delay }) => (
  <Rise delay={delay}>
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: 48,
        borderTop: `1px solid ${theme.rule}`,
        padding: "26px 0",
        width: 1620,
      }}
    >
      <div style={{ flex: "0 0 330px" }}>
        <div
          style={{
            fontFamily: theme.sans,
            fontSize: 42,
            fontWeight: 650,
            color: theme.ink,
            letterSpacing: "-0.02em",
          }}
        >
          {name}
        </div>
        <div
          style={{
            fontFamily: theme.sans,
            fontSize: 24,
            color: theme.ink3,
            marginTop: 8,
          }}
        >
          {what}
        </div>
      </div>
      <div
        style={{
          flex: "0 0 470px",
          fontFamily: theme.serif,
          fontSize: 32,
          color: theme.ink2,
          lineHeight: 1.32,
        }}
      >
        {speed}
      </div>
      <div
        style={{
          flex: "1 1 auto",
          fontFamily: theme.serif,
          fontSize: 32,
          color: theme.ink2,
          lineHeight: 1.32,
        }}
      >
        {reach}
      </div>
    </div>
  </Rise>
);

/* ------------------------------------------------------------ capability  */

const Capability: React.FC<{
  title: string;
  detail: string;
  delay: number;
}> = ({ title, detail, delay }) => (
  <Rise delay={delay}>
    <div
      style={{
        display: "flex",
        gap: 30,
        alignItems: "baseline",
        padding: "20px 0",
        maxWidth: 1600,
      }}
    >
      <div style={{ fontFamily: theme.mono, fontSize: 34, color: theme.ink3 }}>&rarr;</div>
      <div>
        <div
          style={{
            fontFamily: theme.sans,
            fontSize: 46,
            fontWeight: 650,
            letterSpacing: "-0.02em",
            color: theme.ink,
            lineHeight: 1.2,
          }}
        >
          {title}
        </div>
        <div
          style={{
            fontFamily: theme.serif,
            fontSize: 33,
            color: theme.ink2,
            lineHeight: 1.4,
            marginTop: 10,
            maxWidth: 1320,
          }}
        >
          {detail}
        </div>
      </div>
    </div>
  </Rise>
);

/* ----------------------------------------------------------------- video  */

export const Overview: React.FC = () => {
  const at2 = T1;
  const at3 = at2 + T2;
  const at4 = at3 + T3;
  const at5 = at4 + T4;
  const at6 = at5 + T5;
  const at7 = at6 + T6;
  const at8 = at7 + T7;

  return (
    <AbsoluteFill style={{ background: theme.bg }}>
      {/* 1 — the problem, stated without a single technical word */}
      <Sequence durationInFrames={T1}>
        <FadeIO durationInFrames={T1}>
          <Slide wordmark={false}>
            <Kicker delay={0}>Start here</Kicker>
            <div style={{ height: 38 }} />
            <Headline delay={8} size={76}>
              Your brain has about 86 billion cells, and they are all talking at
              once.
            </Headline>
            <div style={{ height: 48 }} />
            <div style={{ width: 900 }}>
              <DrawRule delay={40} />
            </div>
            <div style={{ height: 36 }} />
            <Body delay={48} size={38}>
              Nothing we can put on a person sees more than a sliver of that.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      {/* 2 — and the slivers do not line up */}
      <Sequence from={at2} durationInFrames={T2}>
        <FadeIO durationInFrames={T2}>
          <Slide>
            <Headline delay={0} size={62}>
              Every way of looking sees something different.
            </Headline>
            <div style={{ height: 44 }} />
            <Instrument
              delay={22}
              name="EEG"
              what="electrodes on the scalp"
              speed="Reads a thousand times a second."
              reach="Cannot tell you much about where it came from."
            />
            <Instrument
              delay={46}
              name="MRI"
              what="a scanner around the head"
              speed="Reads once every two seconds."
              reach="Sees the whole brain, blood flow rather than activity."
            />
            <Instrument
              delay={70}
              name="Behaviour"
              what="what the person does"
              speed="Reads whenever they move, look or speak."
              reach="Not a brain measurement at all — but still evidence."
            />
            <div style={{ height: 44 }} />
            <Body delay={96} size={34}>
              Different speeds, different places, different units. And usually
              measured on different people.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      {/* 3 — why the mismatch is the thing that matters */}
      <Sequence from={at3} durationInFrames={T3}>
        <FadeIO durationInFrames={T3}>
          <Slide>
            <Kicker delay={0}>The consequence</Kicker>
            <div style={{ height: 36 }} />
            <Headline delay={8} size={66}>
              So the findings never quite add up — and almost none of them are
              about <em>your</em> brain.
            </Headline>
            <div style={{ height: 44 }} />
            <Body delay={40} size={36}>
              Each study learns a little about an average brain. The average
              brain is a statistic. Nobody is treated with a statistic.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      {/* 4 — the model */}
      <Sequence from={at4} durationInFrames={T4}>
        <FadeIO durationInFrames={T4}>
          <Slide>
            <div style={{ display: "flex", alignItems: "center", gap: 90 }}>
              <div style={{ flex: "1 1 auto" }}>
                <Kicker delay={0}>SC&#8209;WBD</Kicker>
                <div style={{ height: 34 }} />
                <Headline delay={8} size={62}>
                  One working model of one person&#8217;s brain, that every
                  measurement is allowed to correct.
                </Headline>
                <div style={{ height: 40 }} />
                <Body delay={38} size={34}>
                  Not a separate result per instrument. One shared model they all
                  argue with.
                </Body>
              </div>
              <Rise delay={14} distance={0}>
                <ParcelCloud size={560} delay={14} />
              </Rise>
            </div>
          </Slide>
        </FadeIO>
      </Sequence>

      {/* 5 — mechanism, part one: what the model is made of */}
      <Sequence from={at5} durationInFrames={T5}>
        <FadeIO durationInFrames={T5}>
          <Slide>
            <Kicker delay={0}>How it works</Kicker>
            <div style={{ height: 34 }} />
            <Headline delay={6} size={62}>
              The brain is divided into 414 regions, and each one gets its own
              state.
            </Headline>
            <div style={{ height: 50 }} />
            <Rise delay={30}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 24 }}>
                <CountUp to={400} delay={34} duration={30} color={theme.ink} size={92} />
                <span style={{ fontFamily: theme.sans, fontSize: 29, color: theme.ink3 }}>
                  on the surface
                </span>
                <span style={{ fontFamily: theme.mono, fontSize: 58, color: theme.ink3 }}>
                  +
                </span>
                <CountUp to={14} delay={34} duration={30} color={theme.ink} size={92} />
                <span style={{ fontFamily: theme.sans, fontSize: 29, color: theme.ink3 }}>
                  deep inside
                </span>
              </div>
            </Rise>
            <div style={{ height: 46 }} />
            <Body delay={64} size={34}>
              Each region carries its own activity and its own uncertainty, and
              they are wired to each other the way the real ones are.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      {/* 6 — mechanism, part two: the rule that makes partial data usable */}
      <Sequence from={at6} durationInFrames={T6}>
        <FadeIO durationInFrames={T6}>
          <Slide>
            <Kicker delay={0}>The rule that makes it work</Kicker>
            <div style={{ height: 34 }} />
            <Headline delay={6} size={60}>
              Every signal is only allowed to correct the part of the model it
              can actually speak to.
            </Headline>
            <div style={{ height: 46 }} />
            <Body delay={34} size={36}>
              Where you looked is real evidence. It is not a reading of your
              cortex, and a model that confuses the two learns the wrong thing.
            </Body>
            <div style={{ height: 40 }} />
            <div style={{ width: 980 }}>
              <DrawRule delay={64} />
            </div>
            <div style={{ height: 34 }} />
            <Body delay={72} size={36} color={theme.ink}>
              So a person measured only one way still improves the shared model
              — and what was never measured is left unknown rather than invented.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      {/* 7 — what it is for */}
      <Sequence from={at7} durationInFrames={T7}>
        <FadeIO durationInFrames={T7}>
          <Slide>
            <Kicker delay={0}>What you would use it for</Kicker>
            <div style={{ height: 38 }} />
            <Capability
              delay={14}
              title="See what the brain is doing"
              detail="Predict what comes next, and get an honest confidence with it rather than a bare number."
            />
            <Capability
              delay={38}
              title="Try a treatment before it touches anyone"
              detail="Magnetic stimulation is already used for depression, and where you aim it matters. Compare targets on that person's own head, in simulation."
            />
            <Capability
              delay={62}
              title="Tune it to one person"
              detail="Start from the shared model, fit it to an individual, and build a device around the brain in front of you instead of an average one."
            />
          </Slide>
        </FadeIO>
      </Sequence>

      {/* 8 — the status, said plainly, and the way in */}
      <Sequence from={at8} durationInFrames={T8}>
        <FadeIO durationInFrames={T8}>
          <Slide wordmark={false}>
            <Kicker delay={0}>Where it stands today</Kicker>
            <div style={{ height: 34 }} />
            <Headline delay={8} size={58}>
              It does not beat its baselines yet. We publish the numbers anyway.
            </Headline>
            <div style={{ height: 42 }} />
            <Body delay={36} size={36}>
              The weights, the data, the code and the paper are public, including
              every measurement that went against us.
            </Body>
            <div style={{ height: 52 }} />
            <div style={{ width: 760 }}>
              <DrawRule delay={62} />
            </div>
            <div style={{ height: 36 }} />
            <Rise delay={72}>
              <div
                style={{
                  fontFamily: theme.mono,
                  fontSize: 44,
                  letterSpacing: "-0.01em",
                  color: theme.ink,
                }}
              >
                sc-wbd.pages.dev
              </div>
            </Rise>
            <div style={{ height: 18 }} />
            <Rise delay={82}>
              <div
                style={{
                  fontFamily: theme.sans,
                  fontSize: 30,
                  color: theme.ink3,
                }}
              >
                Start with it today.
              </div>
            </Rise>
          </Slide>
        </FadeIO>
      </Sequence>
    </AbsoluteFill>
  );
};
