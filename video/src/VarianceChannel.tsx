import React from "react";
import { AbsoluteFill, Sequence, useCurrentFrame, interpolate } from "remotion";
import { theme } from "./theme";
import {
  Body,
  Code,
  CountUp,
  DrawRule,
  FadeIO,
  Headline,
  Kicker,
  Rise,
  Slide,
  Source,
  Stat,
} from "./components";

/**
 * "The most expensive 134 seconds."
 *
 * Every figure in this composition is traceable to a file in the repository:
 * MSE 3.9697 / 7.1653 and NLL 2.5552 from reports/training/evaluation.json;
 * the excess decomposition and the log_noise statistics from
 * reports/scope_gap.md section 6; the stage-V schedule from
 * scwbd/foundation/train.py:78.
 */

const S1 = 110;
const S2 = 130;
const S3 = 150;
const S4 = 140;
const S5 = 130;
const S6 = 120;

export const VARIANCE_DURATION = S1 + S2 + S3 + S4 + S5 + S6;

/** A two-bar comparison drawn from the actual measured values. */
const BarPair: React.FC<{ delay: number }> = ({ delay }) => {
  const frame = useCurrentFrame();
  const grow = interpolate(frame - delay, [0, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const eased = 1 - Math.pow(1 - grow, 3);
  const rows = [
    { label: "SC-WBD-001-beta", value: 3.9697, color: theme.pass },
    { label: "persistence baseline", value: 7.1653, color: theme.ink3 },
  ];
  const max = 7.1653;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 30, width: 1100 }}>
      {rows.map((r) => (
        <div key={r.label}>
          <div
            style={{
              fontFamily: theme.sans,
              fontSize: 24,
              color: theme.ink2,
              marginBottom: 10,
              display: "flex",
              justifyContent: "space-between",
              width: "100%",
            }}
          >
            <span>{r.label}</span>
            <span style={{ fontFamily: theme.mono, color: r.color }}>{r.value.toFixed(4)}</span>
          </div>
          <div style={{ height: 26, background: theme.bgSoft, borderRadius: 2 }}>
            <div
              style={{
                height: "100%",
                width: `${(r.value / max) * 100 * eased}%`,
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

export const VarianceChannel: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: theme.bg }}>
      <Sequence durationInFrames={S1}>
        <FadeIO durationInFrames={S1}>
          <Slide>
            <Kicker delay={0}>SC-WBD · Engineering</Kicker>
            <div style={{ height: 34 }} />
            <Headline delay={6}>The most expensive 134&nbsp;seconds</Headline>
            <div style={{ height: 40 }} />
            <div style={{ width: 900 }}>
              <DrawRule delay={22} />
            </div>
            <div style={{ height: 40 }} />
            <Body delay={28}>
              A parameter with a closed-form optimum was left to gradient descent
              for two minutes. It cost the entire training run.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={S1} durationInFrames={S2}>
        <FadeIO durationInFrames={S2}>
          <Slide>
            <Kicker delay={0}>The number nobody had quoted</Kicker>
            <div style={{ height: 36 }} />
            <Body delay={8} size={38} color={theme.ink}>
              Mean squared error — lower is better
            </Body>
            <div style={{ height: 44 }} />
            <BarPair delay={20} />
            <div style={{ height: 46 }} />
            <Body delay={60}>
              Our model had the <strong style={{ color: theme.ink }}>lowest MSE of all
              seven arms</strong>. It was reported as beaten by five of six baselines.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={S1 + S2} durationInFrames={S3}>
        <FadeIO durationInFrames={S3}>
          <Slide>
            <Kicker delay={0}>The mechanism</Kicker>
            <div style={{ height: 40 }} />
            <Code delay={8} highlight>
              {`self.log_noise = nn.Parameter(torch.zeros(n_ch))
lv = self.log_noise.expand_as(y)`}
            </Code>
            <div style={{ height: 36 }} />
            <Source delay={24}>scwbd/foundation/heads.py:238</Source>
            <div style={{ height: 44 }} />
            <Body delay={40}>
              The entire predictive variance is one scalar per channel, broadcast.
              <br />
              <strong style={{ color: theme.ink }}>
                lv never reads the state.
              </strong>{" "}
              Constant across time, horizon, window, participant and condition.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={S1 + S2 + S3} durationInFrames={S4}>
        <FadeIO durationInFrames={S4}>
          <Slide>
            <Kicker delay={0}>Decomposing the penalty</Kicker>
            <div style={{ height: 50 }} />
            <div style={{ display: "flex", gap: 190, alignItems: "flex-start" }}>
              <Stat
                delay={10}
                value={<CountUp to={0.4467} decimals={4} delay={14} color={theme.fail} size={112} />}
                label="scale — one global rescale. 100% of the gap."
              />
              <Stat
                delay={26}
                value={<CountUp to={0.0096} decimals={4} delay={30} color={theme.ink3} size={112} />}
                label="horizon — the hypothesis we pre-registered. 2.1%."
              />
            </div>
            <div style={{ height: 56 }} />
            <Body delay={62}>
              The obvious cause was horizon-flatness. It was pre-registered, measured,
              and <strong style={{ color: theme.ink }}>wrong</strong>.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={S1 + S2 + S3 + S4} durationInFrames={S5}>
        <FadeIO durationInFrames={S5}>
          <Slide>
            <Kicker delay={0}>The 134 seconds</Kicker>
            <div style={{ height: 44 }} />
            <div style={{ display: "flex", gap: 90, alignItems: "flex-start" }}>
              <Stat
                delay={8}
                value={<CountUp to={1.379} decimals={3} delay={12} color={theme.ink} size={104} />}
                label="the closed-form optimum"
              />
              <Stat
                delay={20}
                value={<CountUp to={0.273} decimals={3} delay={24} color={theme.fail} size={104} />}
                label="where 900 steps of SGD actually got to"
              />
              <Stat
                delay={32}
                value={<CountUp to={3.0} decimals={1} delay={36} color={theme.fail} size={104} suffix="×" />}
                label="uniformly overconfident, as a result"
              />
            </div>
            <div style={{ height: 56 }} />
            <Body delay={70}>
              Trainable in stage&nbsp;V only. Stage&nbsp;V ran 900 steps at
              lr&nbsp;5.77e-5 — <strong style={{ color: theme.ink }}>134 seconds</strong>.
              About 20% of the way there, still drifting.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={S1 + S2 + S3 + S4 + S5} durationInFrames={S6}>
        <FadeIO durationInFrames={S6}>
          <Slide>
            <Rise delay={0}>
              <div
                style={{
                  fontFamily: theme.serif,
                  fontSize: 62,
                  lineHeight: 1.3,
                  color: theme.ink,
                  fontStyle: "italic",
                  maxWidth: 1400,
                  borderLeft: `3px solid ${theme.ruleStrong}`,
                  paddingLeft: 46,
                }}
              >
                A parameter whose optimum has a closed form was left to SGD for two
                minutes.
              </div>
            </Rise>
            <div style={{ height: 66 }} />
            <Body delay={34} size={32}>
              The forecast was the best in the table. Only the confidence attached to it
              was wrong — and nothing in the instrument could say so.
            </Body>
            <div style={{ height: 40 }} />
            <Rise delay={52}>
              <div style={{ fontFamily: theme.sans, fontSize: 26, color: theme.ink3 }}>
                Full write-up in the engineering essays
              </div>
            </Rise>
          </Slide>
        </FadeIO>
      </Sequence>
    </AbsoluteFill>
  );
};
