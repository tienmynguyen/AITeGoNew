/**
 * SequenceBlender
 * Multi-word chaining and cosine bridge interpolation for continuous VSL sentences.
 */
export class SequenceBlender {
  constructor() {
    this.bridgeFrameCount = 6;
    this.trimBoundaryFrames = 2;
    this.wordMotions = [];      // Array of full word motion JSON objects
    this.blendedFrames = [];    // Compiled flat sequence of frames
    this.currentFrameIdx = 0;
    this.isPlaying = true;
    this.isLooping = true;
    this.playbackSpeed = 1.0;

    // Callbacks
    this.onFrameChange = null;
    this.onWordChange = null;
    this.lastWordIdx = -1;
  }

  setBridgeFrameCount(count) {
    this.bridgeFrameCount = Math.max(1, Math.min(24, Math.round(count)));
    this.rebuildBlendedSequence();
  }

  setPlaybackSpeed(speed) {
    this.playbackSpeed = Math.max(0.2, Math.min(3.0, speed));
  }

  setLooping(loop) {
    this.isLooping = loop;
  }

  /**
   * Set sequence of motions.
   * @param {Array<Object>} motions - Array of motion JSON objects
   */
  setSequence(motions) {
    this.wordMotions = motions.filter(Boolean);
    this.rebuildBlendedSequence();
    this.currentFrameIdx = 0;
    this.lastWordIdx = -1;
  }

  /**
   * Smoothly update sequence mid-playback from current posture.
   * @param {Array<Object>} newMotions
   */
  updateSequenceSmooth(newMotions) {
    if (!newMotions || newMotions.length === 0) {
      this.wordMotions = [];
      this.blendedFrames = [];
      return;
    }

    const currentFrame = this.getCurrentFrame();
    this.wordMotions = newMotions.filter(Boolean);

    if (currentFrame && this.wordMotions.length > 0) {
      // Build new sequence
      const newSequence = this.compileSequence(this.wordMotions);
      if (newSequence.length > 0) {
        // Create dynamic bridge from currentFrame to first frame of new sequence
        const firstTarget = newSequence[0].frame;
        const bridge = this.createBridgeFrames(currentFrame, firstTarget, this.bridgeFrameCount, -1, 'Chuyển tiếp');
        this.blendedFrames = [...bridge, ...newSequence];
        this.currentFrameIdx = 0;
        this.lastWordIdx = -1;
        return;
      }
    }

    this.rebuildBlendedSequence();
    this.currentFrameIdx = 0;
    this.lastWordIdx = -1;
  }

  /**
   * Rebuild the entire blended sequence with bridge frames and boundary trims.
   */
  rebuildBlendedSequence() {
    this.blendedFrames = this.compileSequence(this.wordMotions);
    if (this.currentFrameIdx >= this.blendedFrames.length) {
      this.currentFrameIdx = 0;
    }
  }

  /**
   * Compile a list of motion objects into a smooth sequence.
   */
  compileSequence(motions) {
    if (!motions || motions.length === 0) return [];

    const compiled = [];

    for (let wIdx = 0; wIdx < motions.length; wIdx++) {
      const motion = motions[wIdx];
      const frames = motion.frames || [];
      if (frames.length === 0) continue;

      // Trim 2 redundant idle boundary frames on middle words for seamless continuity
      let startIdx = 0;
      let endIdx = frames.length;

      const isFirst = (wIdx === 0);
      const isLast = (wIdx === motions.length - 1);

      if (!isFirst && frames.length > (this.trimBoundaryFrames * 2 + 2)) {
        startIdx = this.trimBoundaryFrames;
      }
      if (!isLast && (endIdx - startIdx) > (this.trimBoundaryFrames + 2)) {
        endIdx = frames.length - this.trimBoundaryFrames;
      }

      // Append core word frames
      for (let f = startIdx; f < endIdx; f++) {
        compiled.push({
          frame: frames[f],
          wordIndex: wIdx,
          wordName: motion.word,
          isBridge: false,
          bridgeProgress: 0,
          localFrameIndex: f,
          totalWordFrames: frames.length
        });
      }

      // If there is a next word, insert bridge frames
      if (wIdx < motions.length - 1) {
        const nextMotion = motions[wIdx + 1];
        const nextFrames = nextMotion.frames || [];
        if (nextFrames.length > 0) {
          const lastFrameA = frames[endIdx - 1];
          const nextStartIdx = nextFrames.length > (this.trimBoundaryFrames * 2 + 2) ? this.trimBoundaryFrames : 0;
          const firstFrameB = nextFrames[nextStartIdx];

          const bridge = this.createBridgeFrames(
            lastFrameA,
            firstFrameB,
            this.bridgeFrameCount,
            wIdx,
            `${motion.word} -> ${nextMotion.word}`
          );
          compiled.push(...bridge);
        }
      }
    }

    return compiled;
  }

  /**
   * Cosine-interpolated bridge frames between Frame A and Frame B.
   */
  createBridgeFrames(frameA, frameB, count, wordIdx, bridgeLabel) {
    const bridge = [];
    if (!frameA || !frameB || count <= 0) return bridge;

    for (let i = 1; i <= count; i++) {
      const t = i / (count + 1);
      // Cosine interpolation S-curve: zero derivative at endpoints
      const w = (1.0 - Math.cos(Math.PI * t)) / 2.0;

      // 1. Interpolate Pose
      const blendedPose = [];
      const poseA = frameA.pose || [];
      const poseB = frameB.pose || [];
      for (let p = 0; p < 8; p++) {
        const ptA = poseA[p] || [0, 0, 0];
        const ptB = poseB[p] || [0, 0, 0];
        blendedPose.push([
          ptA[0] * (1 - w) + ptB[0] * w,
          ptA[1] * (1 - w) + ptB[1] * w,
          ptA[2] * (1 - w) + ptB[2] * w
        ]);
      }

      // 2. Interpolate Left Hand
      let blendedLeftHand = null;
      if (frameA.leftHand && frameB.leftHand) {
        blendedLeftHand = [];
        for (let h = 0; h < 21; h++) {
          const ptA = frameA.leftHand[h];
          const ptB = frameB.leftHand[h];
          blendedLeftHand.push([
            ptA[0] * (1 - w) + ptB[0] * w,
            ptA[1] * (1 - w) + ptB[1] * w,
            ptA[2] * (1 - w) + ptB[2] * w
          ]);
        }
      } else if (frameA.leftHand && w < 0.6) {
        blendedLeftHand = frameA.leftHand;
      } else if (frameB.leftHand && w >= 0.4) {
        blendedLeftHand = frameB.leftHand;
      }

      // 3. Interpolate Right Hand
      let blendedRightHand = null;
      if (frameA.rightHand && frameB.rightHand) {
        blendedRightHand = [];
        for (let h = 0; h < 21; h++) {
          const ptA = frameA.rightHand[h];
          const ptB = frameB.rightHand[h];
          blendedRightHand.push([
            ptA[0] * (1 - w) + ptB[0] * w,
            ptA[1] * (1 - w) + ptB[1] * w,
            ptA[2] * (1 - w) + ptB[2] * w
          ]);
        }
      } else if (frameA.rightHand && w < 0.6) {
        blendedRightHand = frameA.rightHand;
      } else if (frameB.rightHand && w >= 0.4) {
        blendedRightHand = frameB.rightHand;
      }

      const activeSide = w < 0.5 ? frameA.activeHandSide : frameB.activeHandSide;

      const bridgeFrameData = {
        frameIndex: -1,
        pose: blendedPose,
        leftHand: blendedLeftHand,
        rightHand: blendedRightHand,
        activeHandSide: activeSide
      };

      bridge.push({
        frame: bridgeFrameData,
        wordIndex: wordIdx,
        wordName: bridgeLabel,
        isBridge: true,
        bridgeProgress: t,
        localFrameIndex: i,
        totalWordFrames: count
      });
    }

    return bridge;
  }

  /**
   * Advance playback by deltaTime seconds.
   */
  advance(deltaSeconds) {
    if (!this.isPlaying || this.blendedFrames.length === 0) return;

    const baseFps = 30;
    const framesToAdvance = deltaSeconds * baseFps * this.playbackSpeed;

    this.currentFrameIdx += framesToAdvance;

    if (this.currentFrameIdx >= this.blendedFrames.length) {
      if (this.isLooping) {
        this.currentFrameIdx = this.currentFrameIdx % this.blendedFrames.length;
      } else {
        this.currentFrameIdx = this.blendedFrames.length - 1;
        this.isPlaying = false;
      }
    }

    const currentIntIdx = Math.floor(this.currentFrameIdx);
    const item = this.blendedFrames[currentIntIdx];

    if (item && item.wordIndex !== this.lastWordIdx && !item.isBridge) {
      this.lastWordIdx = item.wordIndex;
      if (this.onWordChange) {
        this.onWordChange(item.wordIndex, item.wordName);
      }
    }

    if (this.onFrameChange) {
      this.onFrameChange(currentIntIdx, this.blendedFrames.length, item);
    }
  }

  getCurrentFrame() {
    const len = this.blendedFrames.length;
    if (len === 0) return null;
    if (len === 1) return this.blendedFrames[0].frame;

    const floatIdx = ((this.currentFrameIdx % len) + len) % len;
    const idxA = Math.floor(floatIdx);
    const idxB = (idxA + 1) % len;
    const alpha = floatIdx - idxA;

    const frameA = this.blendedFrames[idxA].frame;
    const frameB = this.blendedFrames[idxB].frame;

    if (!frameA || !frameB || alpha < 0.001) return frameA;
    if (alpha > 0.999) return frameB;

    // Smoothstep interpolation factor for continuous, stutter-free playback
    const t = alpha * alpha * (3 - 2 * alpha);

    // 1. Interpolate Pose
    const poseA = frameA.pose || [];
    const poseB = frameB.pose || [];
    const pose = [];
    for (let i = 0; i < 8; i++) {
      const pA = poseA[i] || [0, 0, 0];
      const pB = poseB[i] || [0, 0, 0];
      pose.push([
        pA[0] * (1 - t) + pB[0] * t,
        pA[1] * (1 - t) + pB[1] * t,
        pA[2] * (1 - t) + pB[2] * t
      ]);
    }

    // 2. Interpolate Left Hand
    let leftHand = null;
    if (frameA.leftHand && frameB.leftHand) {
      leftHand = [];
      for (let i = 0; i < 21; i++) {
        const hA = frameA.leftHand[i];
        const hB = frameB.leftHand[i];
        leftHand.push([
          hA[0] * (1 - t) + hB[0] * t,
          hA[1] * (1 - t) + hB[1] * t,
          hA[2] * (1 - t) + hB[2] * t
        ]);
      }
    } else {
      leftHand = t < 0.5 ? frameA.leftHand : frameB.leftHand;
    }

    // 3. Interpolate Right Hand
    let rightHand = null;
    if (frameA.rightHand && frameB.rightHand) {
      rightHand = [];
      for (let i = 0; i < 21; i++) {
        const hA = frameA.rightHand[i];
        const hB = frameB.rightHand[i];
        rightHand.push([
          hA[0] * (1 - t) + hB[0] * t,
          hA[1] * (1 - t) + hB[1] * t,
          hA[2] * (1 - t) + hB[2] * t
        ]);
      }
    } else {
      rightHand = t < 0.5 ? frameA.rightHand : frameB.rightHand;
    }

    return {
      frameIndex: idxA,
      pose,
      leftHand,
      rightHand,
      activeHandSide: t < 0.5 ? frameA.activeHandSide : frameB.activeHandSide
    };
  }

  getCurrentItem() {
    if (this.blendedFrames.length === 0) return null;
    const idx = Math.floor(this.currentFrameIdx) % this.blendedFrames.length;
    return this.blendedFrames[idx];
  }

  seek(progressPercent) {
    if (this.blendedFrames.length === 0) return;
    const target = Math.max(0, Math.min(this.blendedFrames.length - 1, Math.floor(progressPercent * this.blendedFrames.length)));
    this.currentFrameIdx = target;
  }

  step(stepDelta) {
    if (this.blendedFrames.length === 0) return;
    this.currentFrameIdx = Math.max(0, Math.min(this.blendedFrames.length - 1, this.currentFrameIdx + stepDelta));
  }

  togglePlay() {
    this.isPlaying = !this.isPlaying;
    return this.isPlaying;
  }

  reset() {
    this.currentFrameIdx = 0;
    this.lastWordIdx = -1;
  }
}

