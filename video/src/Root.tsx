import React from "react";
import { Composition } from "remotion";
import { FPS, HEIGHT, WIDTH } from "./theme";
import { Overview, OVERVIEW_DURATION } from "./Overview";
import { VarianceChannel, VARIANCE_DURATION } from "./VarianceChannel";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="Overview"
        component={Overview}
        durationInFrames={OVERVIEW_DURATION}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
      <Composition
        id="VarianceChannel"
        component={VarianceChannel}
        durationInFrames={VARIANCE_DURATION}
        fps={FPS}
        width={WIDTH}
        height={HEIGHT}
      />
    </>
  );
};
