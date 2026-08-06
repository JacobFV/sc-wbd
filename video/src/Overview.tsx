import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
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
  Stat,
} from "./components";

/**
 * "What SC-WBD is."
 *
 * Constrained by the same rule as the website: no number appears here that is
 * not traceable to a file in the repository. The claim-gate status
 * (all COULD_NOT_RUN) is from reports/CLAIM_BOUNDARY.md; the parcel count from
 * tests/anatomy/test_families.py; the refusal count from scwbd/schema/refusals.py.
 */

const T1 = 120;
const T2 = 150;
const T3 = 140;
const T4 = 130;
const T5 = 120;

export const OVERVIEW_DURATION = T1 + T2 + T3 + T4 + T5;

const Row: React.FC<{ label: string; value: string; delay: number; status?: "pass" | "fail" | "unknown" }> = ({
  label,
  value,
  delay,
  status = "pass",
}) => {
  const color = status === "pass" ? theme.pass : status === "fail" ? theme.fail : theme.unknown;
  return (
    <Rise delay={delay}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 28,
          borderTop: `1px solid ${theme.rule}`,
          padding: "20px 0",
          width: 1420,
        }}
      >
        <div style={{ fontFamily: theme.serif, fontSize: 34, color: theme.ink, flex: "0 0 560px" }}>
          {label}
        </div>
        <div
          style={{
            fontFamily: theme.sans,
            fontSize: 20,
            fontWeight: 600,
            letterSpacing: "0.09em",
            textTransform: "uppercase",
            color,
            border: `1px solid ${color}`,
            borderRadius: 3,
            padding: "3px 12px",
            flex: "0 0 auto",
          }}
        >
          {value}
        </div>
      </div>
    </Rise>
  );
};

export const Overview: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: theme.bg }}>
      <Sequence durationInFrames={T1}>
        <FadeIO durationInFrames={T1}>
          <Slide>
            <Kicker delay={0}>SC-WBD-001 series</Kicker>
            <div style={{ height: 36 }} />
            <Headline delay={6}>
              A whole-brain model you are allowed to disbelieve
            </Headline>
            <div style={{ height: 44 }} />
            <div style={{ width: 1000 }}>
              <DrawRule delay={26} />
            </div>
            <div style={{ height: 44 }} />
            <Body delay={32}>
              Individualised models of a single brain across modalities, scales and
              timescales — and the apparatus for finding out when they are wrong.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={T1} durationInFrames={T2}>
        <FadeIO durationInFrames={T2}>
          <Slide>
            <Kicker delay={0}>What exists today</Kicker>
            <div style={{ height: 44 }} />
            <Row label="Schema and compiler, 11 refusals" value="built" delay={8} />
            <Row label="Anatomy prior, 414 parcels" value="built" delay={18} />
            <Row label="Six dynamics backends + learned control" value="built" delay={28} />
            <Row label="Simulated corpus, 37,888 trajectories" value="built" delay={38} />
            <Row label="Real EEG, 109 participants, 71/11/27" value="built" delay={48} />
            <Row label="TMS impulse-response path" value="partial" delay={58} status="unknown" />
            <Row
              label="A trained model that beats baselines"
              value="does not exist"
              delay={68}
              status="fail"
            />
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={T1 + T2} durationInFrames={T3}>
        <FadeIO durationInFrames={T3}>
          <Slide>
            <Kicker delay={0}>The honest scoreboard</Kicker>
            <div style={{ height: 56 }} />
            <div style={{ display: "flex", gap: 110, alignItems: "flex-start" }}>
              <Stat
                delay={8}
                value={<CountUp to={414} delay={12} color={theme.ink} size={128} />}
                label="parcels in the anatomy prior — 400 cortical, 14 subcortical"
              />
              <Stat
                delay={22}
                value={<CountUp to={11} delay={26} color={theme.ink} size={128} />}
                label="compiler refusals, each with a test that makes it fire"
              />
              <Stat
                delay={36}
                value={<CountUp to={0} delay={40} color={theme.fail} size={128} />}
                label="validated claims about brains — all five claim gates COULD_NOT_RUN"
              />
            </div>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={T1 + T2 + T3} durationInFrames={T4}>
        <FadeIO durationInFrames={T4}>
          <Slide>
            <Kicker delay={0}>Why that is the interesting part</Kicker>
            <div style={{ height: 46 }} />
            <Rise delay={8}>
              <div
                style={{
                  fontFamily: theme.serif,
                  fontSize: 56,
                  lineHeight: 1.32,
                  color: theme.ink,
                  fontStyle: "italic",
                  maxWidth: 1400,
                  borderLeft: `3px solid ${theme.ruleStrong}`,
                  paddingLeft: 46,
                }}
              >
                A constant is the most convincing possible measurement. Every property
                that makes a number trustworthy is maximised by a number that cannot
                move.
              </div>
            </Rise>
            <div style={{ height: 60 }} />
            <Body delay={44} size={32}>
              We trained a model, it lost to five baselines, and the diagnosis was that
              we had built the control arm of our own ablation. Working out why produced
              better material than a win would have.
            </Body>
          </Slide>
        </FadeIO>
      </Sequence>

      <Sequence from={T1 + T2 + T3 + T4} durationInFrames={T5}>
        <FadeIO durationInFrames={T5}>
          <Slide>
            <Headline delay={0} size={64}>
              Every number is traceable to a file.
            </Headline>
            <div style={{ height: 42 }} />
            <Body delay={16}>
              Where a figure could not be traced, there is a question for a human rather
              than a plausible number.
            </Body>
            <div style={{ height: 64 }} />
            <div style={{ width: 700 }}>
              <DrawRule delay={34} />
            </div>
            <div style={{ height: 34 }} />
            <Rise delay={42}>
              <div
                style={{
                  fontFamily: theme.sans,
                  fontSize: 30,
                  letterSpacing: "0.05em",
                  color: theme.ink2,
                }}
              >
                Engineering essays, the paper, and the full claim boundary
              </div>
            </Rise>
          </Slide>
        </FadeIO>
      </Sequence>
    </AbsoluteFill>
  );
};
