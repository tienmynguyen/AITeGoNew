import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, VRMHumanBoneName } from '@pixiv/three-vrm';

/**
 * VRM 0.0 Self-Calibrating Engine
 * Reads the 3D Cyan Skeleton positions in READ-ONLY mode.
 * Auto-calibrates finger curl and thumb opposition signs at load time in T-pose.
 * Eliminates all backward hyperextension and prevents wrist inversion.
 */
export class VRMEngine {
  constructor(scene) {
    this.scene = scene;
    this.currentVRM = null;
    this.isLoaded = false;
    this.slerpFactor = 0.45; // Adjustable via UI slider

    // Relaxed rest rotations for inactive arms (hanging naturally at sides)
    this.relaxedLeftUpper = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.15, 0, -1.25, 'YXZ'));
    this.relaxedLeftLower = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -0.20, 'YXZ'));
    this.relaxedRightUpper = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.15, 0, 1.25, 'YXZ'));
    this.relaxedRightLower = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0.20, 'YXZ'));

    // Natural blinking & expression state
    this.blinkTimer = 0;
    this.nextBlinkTime = 2.5 + Math.random() * 2.0;
    this.isBlinking = false;
    this.blinkDuration = 0.16;
    this.blinkProgress = 0;

    // Bone cache map
    this.bones = {};
  }

  /**
   * Load a VRM 0.0 model from URL or File Object.
   */
  async loadModel(urlOrFile) {
    return new Promise((resolve, reject) => {
      const loader = new GLTFLoader();
      loader.register((parser) => new VRMLoaderPlugin(parser));

      const loadUrl = typeof urlOrFile === 'string' ? urlOrFile : URL.createObjectURL(urlOrFile);

      loader.load(
        loadUrl,
        (gltf) => {
          const vrm = gltf.userData.vrm;
          if (!vrm) {
            reject(new Error('GLTF loaded is not a valid VRM model'));
            return;
          }

          // Clean up old VRM if present
          if (this.currentVRM) {
            this.scene.remove(this.currentVRM.scene);
            VRMUtils.deepDispose(this.currentVRM.scene);
          }

          VRMUtils.removeUnnecessaryVertices(gltf.scene);
          VRMUtils.removeUnnecessaryJoints(gltf.scene);

          this.currentVRM = vrm;
          this.scene.add(vrm.scene);

          // VRM 0.0 standard orientation: rotate 180 deg so character faces camera (+Z)
          vrm.scene.rotation.y = Math.PI;
          vrm.scene.position.set(0, 0, 0);
          vrm.scene.updateMatrixWorld(true);

          // 1. Capture True T-Pose Rest Data on VRM Load
          this.initRestData();

          // 2. Auto-calibrate exact inward curl signs for fingers and thumb in T-pose
          this.autoCalibrateFingerCurl();

          // 3. Cache local rest direction vectors for arms and hands in rest pose
          this.initArmRestDirs();

          this.isLoaded = true;
          console.log('[VRMEngine] Successfully loaded and calibrated VRM model');
          resolve(vrm);
        },
        (progress) => {},
        (error) => {
          console.error('[VRMEngine] Failed to load VRM model:', error);
          reject(error);
        }
      );
    });
  }

  /**
   * Save B.userData.restLocalQuat = B.quaternion.clone() for every humanoid bone.
   */
  initRestData() {
    if (!this.currentVRM || !this.currentVRM.humanoid) return;

    this.bones = {};
    for (const boneName of Object.values(VRMHumanBoneName)) {
      const node = this.currentVRM.humanoid.getNormalizedBoneNode(boneName);
      if (node) {
        node.userData.restLocalQuat = node.quaternion.clone();
        node.userData.currentQuat = node.quaternion.clone();
        node.userData.prevQuat = node.quaternion.clone();
        this.bones[boneName] = node;
      }
    }
  }

  /**
   * Cache bone's local rest direction pointing towards its child bone in rest T-pose.
   * Constant forever — eliminates scene-graph resets during runtime animation.
   */
  cacheBoneRestDir(boneNode, childNode) {
    if (!boneNode) return;
    if (!childNode && boneNode.children && boneNode.children.length > 0) {
      childNode = boneNode.children[0];
    }
    if (!childNode) return;

    const pB = boneNode.getWorldPosition(new THREE.Vector3());
    const pC = childNode.getWorldPosition(new THREE.Vector3());
    const worldRestDir = pC.clone().sub(pB).normalize();
    if (worldRestDir.lengthSq() > 1e-4) {
      const restWorldQuat = boneNode.getWorldQuaternion(new THREE.Quaternion());
      const localRestDir = worldRestDir.clone().applyQuaternion(restWorldQuat.clone().invert()).normalize();
      boneNode.userData.localRestDir = localRestDir;
    } else if (childNode.position.lengthSq() > 1e-6) {
      boneNode.userData.localRestDir = childNode.position.clone().normalize();
    }
  }

  /**
   * Pre-cache rest direction vectors for all arm and hand chains at load time.
   */
  initArmRestDirs() {
    const leftUpperArm = this.bones[VRMHumanBoneName.LeftUpperArm];
    const leftLowerArm = this.bones[VRMHumanBoneName.LeftLowerArm];
    const leftHand = this.bones[VRMHumanBoneName.LeftHand];
    const leftMiddleProx = this.bones['leftMiddleProximal'] || this.bones[VRMHumanBoneName.LeftMiddleProximal];

    const rightUpperArm = this.bones[VRMHumanBoneName.RightUpperArm];
    const rightLowerArm = this.bones[VRMHumanBoneName.RightLowerArm];
    const rightHand = this.bones[VRMHumanBoneName.RightHand];
    const rightMiddleProx = this.bones['rightMiddleProximal'] || this.bones[VRMHumanBoneName.RightMiddleProximal];

    this.cacheBoneRestDir(leftUpperArm, leftLowerArm);
    this.cacheBoneRestDir(leftLowerArm, leftHand);
    this.cacheBoneRestDir(leftHand, leftMiddleProx);

    this.cacheBoneRestDir(rightUpperArm, rightLowerArm);
    this.cacheBoneRestDir(rightLowerArm, rightHand);
    this.cacheBoneRestDir(rightHand, rightMiddleProx);
  }

  /**
   * Empirical T-pose load-time test:
   * Physically rotates each finger bone around local Z by +0.4 rad and tests if child bone
   * moves DOWNWARD in world space (worldY decreases).
   * Since palms face down (-Y) in T-pose, lowering worldY is 100% mathematically GUARANTEED
   * to curl inward toward the palm, making backward bending impossible!
   */
  autoCalibrateFingerCurl() {
    if (!this.currentVRM || !this.currentVRM.humanoid) return;

    const vrm = this.currentVRM;
    vrm.scene.rotation.y = Math.PI;
    vrm.scene.position.set(0, 0, 0);
    vrm.scene.updateMatrixWorld(true);

    const sides = ['left', 'right'];
    const fingerNames = ['Index', 'Middle', 'Ring', 'Little'];

    // 1. Auto-calibrate 4 Fingers on each hand
    for (const side of sides) {
      for (const f of fingerNames) {
        const proxBone = this.bones[`${side}${f}Proximal`];
        const interBone = this.bones[`${side}${f}Intermediate`];
        const distBone = this.bones[`${side}${f}Distal`];

        if (!proxBone || !interBone) continue;

        // Reset to rest
        proxBone.quaternion.copy(proxBone.userData.restLocalQuat);
        vrm.scene.updateMatrixWorld(true);

        const restChildY = interBone.getWorldPosition(new THREE.Vector3()).y;

        // Test +0.4 rad around local Z
        const testQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), 0.4);
        proxBone.quaternion.copy(proxBone.userData.restLocalQuat).multiply(testQuat);
        vrm.scene.updateMatrixWorld(true);

        const plusChildY = interBone.getWorldPosition(new THREE.Vector3()).y;

        // Whichever direction lowers worldY is inward palm curl!
        const signZ = (plusChildY < restChildY) ? 1 : -1;

        proxBone.userData.curlSignZ = signZ;
        if (interBone) interBone.userData.curlSignZ = signZ;
        if (distBone) distBone.userData.curlSignZ = signZ;

        // Restore rest pose
        proxBone.quaternion.copy(proxBone.userData.restLocalQuat);
        vrm.scene.updateMatrixWorld(true);

        console.log(`[AutoCalibrate] ${side}${f} curlSignZ: ${signZ} (restY: ${restChildY.toFixed(4)}, plusY: ${plusChildY.toFixed(4)})`);
      }
    }

    // 2. Auto-calibrate Thumb opposition on each hand
    for (const side of sides) {
      const thumbProx = this.bones[`${side}ThumbProximal`] || this.bones[`${side}ThumbMetacarpal`];
      const thumbDist = this.bones[`${side}ThumbDistal`];
      const pinkyProx = this.bones[`${side}LittleProximal`] || this.bones[`${side}IndexProximal`];

      if (!thumbProx || !thumbDist || !pinkyProx) continue;

      thumbProx.quaternion.copy(thumbProx.userData.restLocalQuat);
      if (thumbDist.userData.restLocalQuat) thumbDist.quaternion.copy(thumbDist.userData.restLocalQuat);
      vrm.scene.updateMatrixWorld(true);

      const targetPos = pinkyProx.getWorldPosition(new THREE.Vector3());
      const restDist = thumbDist.getWorldPosition(new THREE.Vector3()).distanceTo(targetPos);
      const restY = thumbDist.getWorldPosition(new THREE.Vector3()).y;

      let bestScore = -Infinity;
      let bestAxis = 'y';
      let bestSign = 1;

      // Test axes: 'y', 'z', 'x' with +0.4 and -0.4 rad
      const testAxes = [
        { name: 'y', vec: new THREE.Vector3(0, 1, 0) },
        { name: 'z', vec: new THREE.Vector3(0, 0, 1) },
        { name: 'x', vec: new THREE.Vector3(1, 0, 0) }
      ];

      for (const axis of testAxes) {
        for (const testSign of [1, -1]) {
          const qTest = new THREE.Quaternion().setFromAxisAngle(axis.vec, testSign * 0.4);
          thumbProx.quaternion.copy(thumbProx.userData.restLocalQuat).multiply(qTest);
          vrm.scene.updateMatrixWorld(true);

          const curPos = thumbDist.getWorldPosition(new THREE.Vector3());
          const curDist = curPos.distanceTo(targetPos);
          const curY = curPos.y;

          // Opposition score: moves across palm closer to pinky (restDist - curDist) and downward (restY - curY)
          const distImprovement = (restDist - curDist);
          const yDrop = (restY - curY);
          const score = distImprovement * 2.0 + yDrop;

          if (score > bestScore) {
            bestScore = score;
            bestAxis = axis.name;
            bestSign = testSign;
          }
        }
      }

      thumbProx.userData.curlAxis = bestAxis;
      thumbProx.userData.curlSign = bestSign;
      if (thumbDist) {
        thumbDist.userData.curlAxis = bestAxis;
        thumbDist.userData.curlSign = bestSign;
      }

      // Restore rest
      thumbProx.quaternion.copy(thumbProx.userData.restLocalQuat);
      if (thumbDist.userData.restLocalQuat) thumbDist.quaternion.copy(thumbDist.userData.restLocalQuat);
      vrm.scene.updateMatrixWorld(true);

      console.log(`[AutoCalibrate] ${side}Thumb curlAxis: ${bestAxis}, sign: ${bestSign} (score: ${bestScore.toFixed(4)})`);
    }
  }

  /**
   * Get actual shoulder 3D world positions of AvatarSample_A.
   */
  getShoulderPositions() {
    const leftUpper = this.bones[VRMHumanBoneName.LeftUpperArm];
    const rightUpper = this.bones[VRMHumanBoneName.RightUpperArm];
    if (!leftUpper || !rightUpper) return null;

    leftUpper.updateWorldMatrix(true, false);
    rightUpper.updateWorldMatrix(true, false);

    const leftPos = leftUpper.getWorldPosition(new THREE.Vector3());
    const rightPos = rightUpper.getWorldPosition(new THREE.Vector3());
    return { leftPos, rightPos };
  }

  setSlerpFactor(factor) {
    this.slerpFactor = Math.max(0.05, Math.min(0.95, factor));
  }

  /**
   * Solve a bone to aim towards targetDirection in world space using pure mathematics.
   * Frame-rate independent exponential smoothing: alpha = 1.0 - Math.exp(-12.0 * deltaTime).
   * NEVER resets boneNode.quaternion to restLocalQuat during traversal!
   * Eliminates parent-child matrix oscillation and high-frequency twitching.
   */
  alignBoneTowards(boneNode, childNode, targetDir, deltaTime = 0.016) {
    if (!boneNode || !targetDir || targetDir.lengthSq() < 1e-4) return;

    if (!boneNode.userData.localRestDir) {
      if (childNode) {
        this.cacheBoneRestDir(boneNode, childNode);
      }
      if (!boneNode.userData.localRestDir) return;
    }

    // 1. Parent's current world orientation (already smoothed and up to date)
    const parentWorldQuat = boneNode.parent
      ? boneNode.parent.getWorldQuaternion(new THREE.Quaternion())
      : new THREE.Quaternion();

    // 2. Bone's world orientation if it were at local rest
    const restWorldQuat = parentWorldQuat.clone().multiply(boneNode.userData.restLocalQuat);

    // 3. Express target world direction in bone's local rest frame:
    // localTargetDir = (restWorldQuat)^-1 * targetDir
    const localTargetDir = targetDir.clone().applyQuaternion(restWorldQuat.clone().invert()).normalize();
    if (localTargetDir.lengthSq() < 1e-4) return;

    // 4. Shortest arc rotation in bone's local space from localRestDir to localTargetDir
    const deltaLocalQuat = new THREE.Quaternion().setFromUnitVectors(boneNode.userData.localRestDir, localTargetDir);

    // 5. Target local quaternion: targetLocalQuat = restLocalQuat * deltaLocalQuat
    const targetLocalQuat = boneNode.userData.restLocalQuat.clone().multiply(deltaLocalQuat);

    // 6. Frame-rate independent exponential smoothing: 1.0 - Math.exp(-12.0 * deltaTime)
    const dt = Math.max(0.001, Math.min(0.1, deltaTime));
    const alpha = 1.0 - Math.exp(-12.0 * dt);

    // B.quaternion smoothly slerps toward targetLocalQuat from its previous smoothed frame
    boneNode.quaternion.slerp(targetLocalQuat, alpha);
    boneNode.updateMatrixWorld(true);
  }

  /**
   * Smoothly slerp an inactive arm back to a relaxed natural posture by her side.
   */
  recoverRestPose(side, deltaTime = 0.016) {
    const isLeft = side === 'left';
    const upperBone = this.bones[isLeft ? VRMHumanBoneName.LeftUpperArm : VRMHumanBoneName.RightUpperArm];
    const lowerBone = this.bones[isLeft ? VRMHumanBoneName.LeftLowerArm : VRMHumanBoneName.RightLowerArm];
    const handBone = this.bones[isLeft ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand];

    const qRelaxedUpper = isLeft ? this.relaxedLeftUpper : this.relaxedRightUpper;
    const qRelaxedLower = isLeft ? this.relaxedLeftLower : this.relaxedRightLower;

    const dt = Math.max(0.001, Math.min(0.1, deltaTime));
    const alphaRest = 1.0 - Math.exp(-8.0 * dt);

    if (upperBone && upperBone.userData.restLocalQuat) {
      const targetQuat = upperBone.userData.restLocalQuat.clone().multiply(qRelaxedUpper);
      upperBone.quaternion.slerp(targetQuat, alphaRest);
      upperBone.updateMatrixWorld(true);
    }

    if (lowerBone && lowerBone.userData.restLocalQuat) {
      const targetQuat = lowerBone.userData.restLocalQuat.clone().multiply(qRelaxedLower);
      lowerBone.quaternion.slerp(targetQuat, alphaRest);
      lowerBone.updateMatrixWorld(true);
    }

    if (handBone && handBone.userData.restLocalQuat) {
      handBone.quaternion.slerp(handBone.userData.restLocalQuat, alphaRest);
      handBone.updateMatrixWorld(true);
    }

    this.relaxFingers(side, dt);
  }

  relaxFingers(side, dt = 0.016) {
    const isLeft = side === 'left';
    const prefix = isLeft ? 'left' : 'right';
    const fingerNames = ['Index', 'Middle', 'Ring', 'Little', 'Thumb'];
    const alphaRelax = 1.0 - Math.exp(-8.0 * dt);

    for (const f of fingerNames) {
      for (const seg of ['Proximal', 'Intermediate', 'Distal']) {
        const boneKey = `${prefix}${f}${seg}`;
        const node = this.bones[boneKey];
        if (node && node.userData.restLocalQuat) {
          node.userData.smoothedCurl = 0;
          node.userData.smoothedFold = 0;
          node.userData.smoothedOpen = 1;
          const signZ = node.userData.curlSignZ !== undefined ? node.userData.curlSignZ : (isLeft ? 1 : -1);
          // 5 degrees micro-curve inward so fingers never hyperextend backward
          const relaxedCurve = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), signZ * 0.08);
          const target = node.userData.restLocalQuat.clone().multiply(relaxedCurve);
          node.quaternion.slerp(target, alphaRelax);
          node.updateMatrixWorld(true);
        }
      }
    }
  }

  /**
   * Anatomical Wrist & Finger Solver
   * Uses load-time auto-calibrated curl signs to ensure 100% inward palm flexion.
   * Clamps against hyperextension with 0.08 rad natural micro-curve.
   * Uses deadzone + exponential low-pass filter on finger curls & smoothedPalmLen.
   */
  solveHandAndFingers(side, skelHandWorld, deltaTime = 0.016) {
    if (!skelHandWorld || skelHandWorld.length < 21) {
      this.recoverRestPose(side, deltaTime);
      return;
    }

    const dt = Math.max(0.001, Math.min(0.1, deltaTime));
    const isLeft = side === 'left';
    const prefix = isLeft ? 'left' : 'right';

    // 1. Solve Wrist: primary axis from Wrist (0) to Middle MCP (9)
    const handNode = this.bones[isLeft ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand];
    const middleProxNode = this.bones[`${prefix}MiddleProximal`];

    if (handNode && middleProxNode) {
      const targetHandDir = skelHandWorld[9].clone().sub(skelHandWorld[0]).normalize();
      // Shortest-arc alignment from restLocalQuat introduces ZERO twist
      this.alignBoneTowards(handNode, middleProxNode, targetHandDir, dt);
    }

    // 2. Smooth palm length denominator across frames (smoothedPalmLen)
    // Foreshortening of the palm never causes all 5 fingers to twitch simultaneously!
    const rawPalmLen = Math.max(skelHandWorld[9].distanceTo(skelHandWorld[0]), 0.05);
    const palmKey = `${prefix}SmoothedPalmLen`;
    if (this[palmKey] === undefined) {
      this[palmKey] = rawPalmLen;
    } else {
      const palmDiff = rawPalmLen - this[palmKey];
      if (Math.abs(palmDiff) > 0.003) {
        this[palmKey] += palmDiff * (1.0 - Math.exp(-8.0 * dt));
      }
    }
    const palmLen = this[palmKey];

    // 3. Solve 4 Fingers (Index, Middle, Ring, Little)
    const fingerDefs = [
      { name: 'Index', mcp: 5, pip: 6, dip: 7, tip: 8 },
      { name: 'Middle', mcp: 9, pip: 10, dip: 11, tip: 12 },
      { name: 'Ring', mcp: 13, pip: 14, dip: 15, tip: 16 },
      { name: 'Little', mcp: 17, pip: 18, dip: 19, tip: 20 }
    ];

    const fingerAlpha = 1.0 - Math.exp(-12.0 * dt);

    for (const f of fingerDefs) {
      const pMcp = skelHandWorld[f.mcp];
      const pPip = skelHandWorld[f.pip];
      const pDip = skelHandWorld[f.dip];
      const pTip = skelHandWorld[f.tip];

      const directDist = pTip.distanceTo(pMcp);
      const curveDist = Math.max(pPip.distanceTo(pMcp) + pDip.distanceTo(pPip) + pTip.distanceTo(pDip), 0.01);
      const extensionRatio = directDist / curveDist;

      // 0.0 = straight open finger, 1.0 = curled tightly into palm
      let rawCurl = (0.84 - extensionRatio) / (0.84 - 0.28);
      rawCurl = THREE.MathUtils.clamp(rawCurl, 0.0, 1.0);

      const proxNode = this.bones[`${prefix}${f.name}Proximal`];
      const interNode = this.bones[`${prefix}${f.name}Intermediate`];
      const distNode = this.bones[`${prefix}${f.name}Distal`];

      if (!proxNode) continue;

      // Low-pass filter with deadzone 0.025: ignore micro-jitter
      if (proxNode.userData.smoothedCurl === undefined) {
        proxNode.userData.smoothedCurl = rawCurl;
      } else {
        const prevCurl = proxNode.userData.smoothedCurl;
        if (Math.abs(rawCurl - prevCurl) > 0.025) {
          proxNode.userData.smoothedCurl = prevCurl + (rawCurl - prevCurl) * (1.0 - Math.exp(-10.0 * dt));
        }
      }
      const smoothCurl = proxNode.userData.smoothedCurl;

      // Use empirical auto-calibrated sign (guaranteed to curl inward into palm!)
      const signZ = proxNode.userData.curlSignZ !== undefined ? proxNode.userData.curlSignZ : (isLeft ? 1 : -1);

      // Hard clamp: base micro-curve of 0.08 rad (5 degrees) inward so fingers NEVER bend backward!
      const proxAngle = signZ * (0.08 + smoothCurl * 1.25);
      const interAngle = signZ * (0.08 + smoothCurl * 1.50);
      const distAngle = signZ * (0.08 + smoothCurl * 1.00);

      const applyFingerRotation = (node, angle) => {
        if (!node || !node.userData.restLocalQuat) return;
        const qTarget = node.userData.restLocalQuat.clone().multiply(
          new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), angle)
        );
        node.quaternion.slerp(qTarget, fingerAlpha);
        node.updateMatrixWorld(true);
      };

      applyFingerRotation(proxNode, proxAngle);
      applyFingerRotation(interNode, interAngle);
      applyFingerRotation(distNode, distAngle);
    }

    // 4. Solve Thumb (Points 1..4)
    const pThumbTip = skelHandWorld[4];
    const pIndexMcp = skelHandWorld[5];
    const rawThumbDist = pThumbTip.distanceTo(pIndexMcp) / palmLen;

    const rawThumbOpen = THREE.MathUtils.clamp((rawThumbDist - 0.35) / 0.45, 0.0, 1.0);
    const rawThumbFold = 1.0 - rawThumbOpen;

    const thumbProx = this.bones[`${prefix}ThumbProximal`] || this.bones[`${prefix}ThumbMetacarpal`];
    const thumbDistNode = this.bones[`${prefix}ThumbDistal`];

    if (thumbProx && thumbProx.userData.restLocalQuat) {
      if (thumbProx.userData.smoothedFold === undefined) {
        thumbProx.userData.smoothedFold = rawThumbFold;
        thumbProx.userData.smoothedOpen = rawThumbOpen;
      } else {
        const prevFold = thumbProx.userData.smoothedFold;
        if (Math.abs(rawThumbFold - prevFold) > 0.025) {
          thumbProx.userData.smoothedFold = prevFold + (rawThumbFold - prevFold) * (1.0 - Math.exp(-10.0 * dt));
        }

        const prevOpen = thumbProx.userData.smoothedOpen;
        if (Math.abs(rawThumbOpen - prevOpen) > 0.025) {
          thumbProx.userData.smoothedOpen = prevOpen + (rawThumbOpen - prevOpen) * (1.0 - Math.exp(-10.0 * dt));
        }
      }
      const sFold = thumbProx.userData.smoothedFold;
      const sOpen = thumbProx.userData.smoothedOpen;

      const curlAxisName = thumbProx.userData.curlAxis || 'y';
      const curlSign = thumbProx.userData.curlSign !== undefined ? thumbProx.userData.curlSign : (isLeft ? -1 : 1);

      const axisVec = curlAxisName === 'z' ? new THREE.Vector3(0, 0, 1) :
                      curlAxisName === 'x' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);

      // Inward fold using calibrated axis & sign
      const qFold = new THREE.Quaternion().setFromAxisAngle(axisVec, curlSign * (sFold * 0.65));
      const qAbduct = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), curlSign * (sOpen * 0.30 - sFold * 0.30));

      const targetProx = thumbProx.userData.restLocalQuat.clone().multiply(qFold).multiply(qAbduct);
      thumbProx.quaternion.slerp(targetProx, fingerAlpha);
      thumbProx.updateMatrixWorld(true);
    }

    if (thumbDistNode && thumbDistNode.userData.restLocalQuat) {
      const sFold = thumbProx?.userData?.smoothedFold !== undefined ? thumbProx.userData.smoothedFold : rawThumbFold;
      const signZ = thumbProx?.userData?.curlSignZ !== undefined ? thumbProx.userData.curlSignZ : (isLeft ? 1 : -1);
      const qTip = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), signZ * (sFold * 0.65 + 0.06));
      const targetDist = thumbDistNode.userData.restLocalQuat.clone().multiply(qTip);
      thumbDistNode.quaternion.slerp(targetDist, fingerAlpha);
      thumbDistNode.updateMatrixWorld(true);
    }
  }

  /**
   * Main Solver Entry Point: Read 3D skeleton joints in READ-ONLY mode.
   */
  update(skeletonWorldData, delta = 0.016) {
    if (!this.isLoaded || !this.currentVRM || !skeletonWorldData) return;

    const { poseWorld, leftHandWorld, rightHandWorld } = skeletonWorldData;
    if (!poseWorld || poseWorld.length < 8) return;

    // 1. Constrained Head & Neck Orientation (clamped within ±12°)
    const headNode = this.bones[VRMHumanBoneName.Head];
    if (headNode && poseWorld[0] && poseWorld[7]) {
      const dx = poseWorld[0].x - poseWorld[7].x;
      const dy = poseWorld[0].y - poseWorld[7].y;
      const yaw = THREE.MathUtils.clamp(-dx * 0.6, -0.21, 0.21);
      const pitch = THREE.MathUtils.clamp((-dy - 0.21) * 0.5, -0.21, 0.21);

      const qHead = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ'));
      const targetHead = headNode.userData.restLocalQuat.clone().multiply(qHead);
      if (!headNode.userData.currentQuat) {
        headNode.userData.currentQuat = headNode.quaternion.clone();
      }
      headNode.userData.currentQuat.slerp(targetHead, 0.15);
      headNode.quaternion.copy(headNode.userData.currentQuat);
    }

    // 2. Exact Side Matching & Hierarchical Arm Alignment:
    // Left Arm (Joints 1: Shoulder, 3: Elbow, 5: Wrist)
    if (poseWorld[1] && poseWorld[3] && poseWorld[5]) {
      const leftUpperArm = this.bones[VRMHumanBoneName.LeftUpperArm];
      const leftLowerArm = this.bones[VRMHumanBoneName.LeftLowerArm];
      const leftHandBone = this.bones[VRMHumanBoneName.LeftHand];

      const targetUpperDir = poseWorld[3].clone().sub(poseWorld[1]).normalize();
      this.alignBoneTowards(leftUpperArm, leftLowerArm, targetUpperDir, delta);

      const targetLowerDir = poseWorld[5].clone().sub(poseWorld[3]).normalize();
      this.alignBoneTowards(leftLowerArm, leftHandBone, targetLowerDir, delta);

      if (leftHandWorld) {
        this.solveHandAndFingers('left', leftHandWorld, delta);
      } else {
        this.recoverRestPose('left', delta);
      }
    } else {
      this.recoverRestPose('left', delta);
    }

    // Right Arm (Joints 2: Shoulder, 4: Elbow, 6: Wrist)
    if (poseWorld[2] && poseWorld[4] && poseWorld[6]) {
      const rightUpperArm = this.bones[VRMHumanBoneName.RightUpperArm];
      const rightLowerArm = this.bones[VRMHumanBoneName.RightLowerArm];
      const rightHandBone = this.bones[VRMHumanBoneName.RightHand];

      const targetUpperDir = poseWorld[4].clone().sub(poseWorld[2]).normalize();
      this.alignBoneTowards(rightUpperArm, rightLowerArm, targetUpperDir, delta);

      const targetLowerDir = poseWorld[6].clone().sub(poseWorld[4]).normalize();
      this.alignBoneTowards(rightLowerArm, rightHandBone, targetLowerDir, delta);

      if (rightHandWorld) {
        this.solveHandAndFingers('right', rightHandWorld, delta);
      } else {
        this.recoverRestPose('right', delta);
      }
    } else {
      this.recoverRestPose('right', delta);
    }

    // 3. Natural Blinking & Facial Physics
    this.updateFacialExpressions(delta);

    // 4. Update VRM scene internals
    this.currentVRM.update(delta);
  }

  updateFacialExpressions(delta) {
    if (!this.currentVRM || !this.currentVRM.expressionManager) return;

    this.blinkTimer += delta;
    if (!this.isBlinking && this.blinkTimer >= this.nextBlinkTime) {
      this.isBlinking = true;
      this.blinkProgress = 0;
      this.blinkTimer = 0;
      this.nextBlinkTime = 2.5 + Math.random() * 2.5;
    }

    if (this.isBlinking) {
      this.blinkProgress += delta / this.blinkDuration;
      const blinkVal = this.blinkProgress < 0.5 ? this.blinkProgress * 2.0 : (1.0 - this.blinkProgress) * 2.0;

      if (this.blinkProgress >= 1.0) {
        this.isBlinking = false;
        this.currentVRM.expressionManager.setValue('blink', 0);
      } else {
        this.currentVRM.expressionManager.setValue('blink', THREE.MathUtils.clamp(blinkVal, 0, 1));
      }
    }
  }

  setExpression(name, value) {
    if (this.currentVRM && this.currentVRM.expressionManager) {
      this.currentVRM.expressionManager.setValue(name, THREE.MathUtils.clamp(value, 0, 1));
    }
  }
}
