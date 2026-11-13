/**
 * NpyParser
 * In-browser binary parser for .npy files with client-side 3D depth synthesis.
 */

export function parseNpyBuffer(arrayBuffer, filename = 'custom.npy') {
  const dataView = new DataView(arrayBuffer);

  // 1. Verify magic bytes: \x93NUMPY
  const magic = String.fromCharCode(
    dataView.getUint8(0),
    dataView.getUint8(1),
    dataView.getUint8(2),
    dataView.getUint8(3),
    dataView.getUint8(4),
    dataView.getUint8(5)
  );

  if (magic !== '\x93NUMPY') {
    throw new Error('Not a valid .npy binary file');
  }

  const majorVersion = dataView.getUint8(6);
  const minorVersion = dataView.getUint8(7);

  let headerLength = 0;
  let offset = 8;
  if (majorVersion === 1) {
    headerLength = dataView.getUint16(offset, true);
    offset += 2;
  } else if (majorVersion === 2) {
    headerLength = dataView.getUint32(offset, true);
    offset += 4;
  } else {
    throw new Error(`Unsupported .npy major version: ${majorVersion}`);
  }

  // Read header string (ASCII / UTF-8 dictionary)
  const headerBytes = new Uint8Array(arrayBuffer, offset, headerLength);
  const headerStr = new TextDecoder('utf-8').decode(headerBytes);
  offset += headerLength;

  // Extract shape from header: e.g. 'shape': (30, 58), or (58,)
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]+)\)/);
  if (!shapeMatch) {
    throw new Error('Could not parse shape from .npy header');
  }

  const shapeParts = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);

  let framesCount = 1;
  let D = 0;

  if (shapeParts.length === 1) {
    framesCount = 1;
    D = shapeParts[0];
  } else if (shapeParts.length >= 2) {
    framesCount = shapeParts[0];
    D = shapeParts[1];
  }

  // Read Float32 data
  const floatCount = framesCount * D;
  const floatArray = new Float32Array(arrayBuffer, offset, floatCount);

  // If 1-frame static motion, replicate to 30 frames for smooth playback
  const targetFrames = framesCount === 1 ? 30 : framesCount;
  const wordName = filename.replace(/\.npy$/i, '');

  let dimMode = 'one_hand_2d';
  if (D === 58) dimMode = 'one_hand_2d';
  else if (D === 87) dimMode = 'one_hand_3d';
  else if (D === 100) dimMode = 'two_hands_2d';
  else if (D === 150) dimMode = 'two_hands_3d';
  else if (D >= 1629) dimMode = 'mediapipe_1662';
  else dimMode = D % 2 === 0 ? 'one_hand_2d' : 'one_hand_3d';

  const frames = [];
  for (let f = 0; f < targetFrames; f++) {
    const srcF = framesCount === 1 ? 0 : f;
    const frameSlice = floatArray.subarray(srcF * D, (srcF + 1) * D);
    const frameObj = processFrameSlice(frameSlice, dimMode, f);
    frames.push(frameObj);
  }

  return {
    word: wordName,
    category: dimMode.includes('one_hand') ? 'one_hand' : 'two_hands',
    mode: dimMode.includes('one_hand') ? 'one_hand' : 'two_hands',
    fps: 30,
    frameCount: targetFrames,
    hasDepth: true,
    sourceShape: [framesCount, D],
    frames
  };
}

function processFrameSlice(slice, dimMode, frameIndex) {
  if (dimMode === 'one_hand_2d' || dimMode === 'one_hand_3d') {
    const is3d = dimMode === 'one_hand_3d';
    const stride = is3d ? 3 : 2;

    const pose = [];
    for (let i = 0; i < 8; i++) {
      const idx = i * stride;
      const x = slice[idx];
      const y = slice[idx + 1];
      const z = is3d ? slice[idx + 2] : 0.0;
      pose.push([x, y, z]);
    }

    if (!is3d) {
      pose[0][2] = -0.06; // Nose slightly forward
      const L_upper = 0.95;
      const L_lower = 0.90;

      // Left Arm forward kinematic depth
      const dL_upper = Math.hypot(pose[3][0] - pose[1][0], pose[3][1] - pose[1][1]);
      const dL_lower = Math.hypot(pose[5][0] - pose[3][0], pose[5][1] - pose[3][1]);
      if (pose[5][1] < 0.6 || Math.abs(pose[5][0]) < 0.65) {
        let e_z = Math.max(0.06, Math.sqrt(Math.max(0, L_upper * L_upper - dL_upper * dL_upper)) * 0.65);
        let w_z = e_z + Math.max(0.10, Math.sqrt(Math.max(0, L_lower * L_lower - dL_lower * dL_lower)) * 0.85);
        if (Math.abs(pose[5][0]) < 0.65 && pose[5][1] < 0.6) {
          e_z = Math.max(e_z, 0.14);
          w_z = Math.max(w_z, 0.22);
        }
        pose[3][2] = -e_z;
        pose[5][2] = -w_z;
      }

      // Right Arm forward kinematic depth
      const dR_upper = Math.hypot(pose[4][0] - pose[2][0], pose[4][1] - pose[2][1]);
      const dR_lower = Math.hypot(pose[6][0] - pose[4][0], pose[6][1] - pose[4][1]);
      if (pose[6][1] < 0.6 || Math.abs(pose[6][0]) < 0.65) {
        let e_z = Math.max(0.06, Math.sqrt(Math.max(0, L_upper * L_upper - dR_upper * dR_upper)) * 0.65);
        let w_z = e_z + Math.max(0.10, Math.sqrt(Math.max(0, L_lower * L_lower - dR_lower * dR_lower)) * 0.85);
        if (Math.abs(pose[6][0]) < 0.65 && pose[6][1] < 0.6) {
          e_z = Math.max(e_z, 0.14);
          w_z = Math.max(w_z, 0.22);
        }
        pose[4][2] = -e_z;
        pose[6][2] = -w_z;
      }
    }

    const hand = [];
    const handStart = 8 * stride;
    const handWristX = slice[handStart];
    const handWristY = slice[handStart + 1];

    const dL = Math.hypot(handWristX - pose[5][0], handWristY - pose[5][1]);
    const dR = Math.hypot(handWristX - pose[6][0], handWristY - pose[6][1]);
    const activeSide = dL <= dR ? 'leftHand' : 'rightHand';
    const wristZ = activeSide === 'leftHand' ? pose[5][2] : pose[6][2];

    const palmLen = Math.hypot(
      slice[handStart + 9 * stride] - handWristX,
      slice[handStart + 9 * stride + 1] - handWristY
    ) || 0.15;
    const scaleRatio = palmLen / 0.15;

    const mcpArch = { 1: -0.025 * scaleRatio, 2: -0.040 * scaleRatio, 5: -0.015 * scaleRatio, 9: -0.022 * scaleRatio, 13: -0.015 * scaleRatio, 17: -0.005 * scaleRatio };

    for (let i = 0; i < 21; i++) {
      const idx = handStart + i * stride;
      const x = slice[idx];
      const y = slice[idx + 1];
      let z = is3d ? slice[idx + 2] : wristZ;
      if (!is3d && i > 0 && mcpArch[i]) {
        z += mcpArch[i];
      }
      hand.push([x, y, z]);
    }

    // For single-hand motion, ensure inactive arm is straight down below its shoulder
    if (activeSide === 'leftHand') {
      pose[4] = [pose[2][0], pose[2][1] + 0.95, 0.0];
      pose[6] = [pose[2][0], pose[2][1] + 1.85, 0.0];
    } else {
      pose[3] = [pose[1][0], pose[1][1] + 0.95, 0.0];
      pose[5] = [pose[1][0], pose[1][1] + 1.85, 0.0];
    }

    return {
      frameIndex,
      pose,
      activeHandSide: activeSide,
      leftHand: activeSide === 'leftHand' ? hand : null,
      rightHand: activeSide === 'rightHand' ? hand : null
    };
  }

  // Two hands mode
  const is3d = dimMode === 'two_hands_3d';
  const stride = is3d ? 3 : 2;

  const pose = [];
  for (let i = 0; i < 8; i++) {
    const idx = i * stride;
    const x = slice[idx];
    const y = slice[idx + 1];
    const z = is3d ? slice[idx + 2] : 0.0;
    pose.push([x, y, z]);
  }

  if (!is3d) {
    pose[0][2] = -0.06;
    const L_upper = 0.95;
    const L_lower = 0.90;

    // Left Arm forward kinematic depth
    const dL_upper = Math.hypot(pose[3][0] - pose[1][0], pose[3][1] - pose[1][1]);
    const dL_lower = Math.hypot(pose[5][0] - pose[3][0], pose[5][1] - pose[3][1]);
    if (pose[5][1] < 0.6 || Math.abs(pose[5][0]) < 0.65) {
      let e_z = Math.max(0.06, Math.sqrt(Math.max(0, L_upper * L_upper - dL_upper * dL_upper)) * 0.65);
      let w_z = e_z + Math.max(0.10, Math.sqrt(Math.max(0, L_lower * L_lower - dL_lower * dL_lower)) * 0.85);
      if (Math.abs(pose[5][0]) < 0.65 && pose[5][1] < 0.6) {
        e_z = Math.max(e_z, 0.14);
        w_z = Math.max(w_z, 0.22);
      }
      pose[3][2] = -e_z;
      pose[5][2] = -w_z;
    }

    // Right Arm forward kinematic depth
    const dR_upper = Math.hypot(pose[4][0] - pose[2][0], pose[4][1] - pose[2][1]);
    const dR_lower = Math.hypot(pose[6][0] - pose[4][0], pose[6][1] - pose[4][1]);
    if (pose[6][1] < 0.6 || Math.abs(pose[6][0]) < 0.65) {
      let e_z = Math.max(0.06, Math.sqrt(Math.max(0, L_upper * L_upper - dR_upper * dR_upper)) * 0.65);
      let w_z = e_z + Math.max(0.10, Math.sqrt(Math.max(0, L_lower * L_lower - dR_lower * dR_lower)) * 0.85);
      if (Math.abs(pose[6][0]) < 0.65 && pose[6][1] < 0.6) {
        e_z = Math.max(e_z, 0.14);
        w_z = Math.max(w_z, 0.22);
      }
      pose[4][2] = -e_z;
      pose[6][2] = -w_z;
    }
  }

  // Left hand: points 8..28
  const lhStart = 8 * stride;
  let lhNorm = 0;
  for (let i = 0; i < 21 * stride; i++) lhNorm += Math.abs(slice[lhStart + i]);

  let leftHand = null;
  if (lhNorm > 1e-4) {
    leftHand = [];
    const wristZ = pose[5][2];
    for (let i = 0; i < 21; i++) {
      const idx = lhStart + i * stride;
      leftHand.push([slice[idx], slice[idx + 1], is3d ? slice[idx + 2] : wristZ - 0.02]);
    }
  }

  // Right hand: points 29..49
  const rhStart = 29 * stride;
  let rhNorm = 0;
  for (let i = 0; i < 21 * stride; i++) rhNorm += Math.abs(slice[rhStart + i]);

  let rightHand = null;
  if (rhNorm > 1e-4) {
    rightHand = [];
    const wristZ = pose[6][2];
    for (let i = 0; i < 21; i++) {
      const idx = rhStart + i * stride;
      rightHand.push([slice[idx], slice[idx + 1], is3d ? slice[idx + 2] : wristZ - 0.02]);
    }
  }

  let activeSide = 'none';
  if (leftHand && rightHand) activeSide = 'both';
  else if (leftHand) activeSide = 'leftHand';
  else if (rightHand) activeSide = 'rightHand';

  return {
    frameIndex,
    pose,
    activeHandSide: activeSide,
    leftHand,
    rightHand
  };
}

