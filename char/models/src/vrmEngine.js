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
    this.relaxedLeftUpper = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.02, 0, -1.48, 'YXZ'));
    this.relaxedLeftLower = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0, 'YXZ'));
    this.relaxedRightUpper = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.02, 0, 1.48, 'YXZ'));
    this.relaxedRightLower = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0, 'YXZ'));

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

    // 2. Auto-calibrate Thumb opposition & flexion on each hand
    for (const side of sides) {
      const thumbProx = this.bones[`${side}ThumbProximal`] || this.bones[`${side}ThumbMetacarpal`];
      const thumbInter = this.bones[`${side}ThumbIntermediate`];
      const thumbDist = this.bones[`${side}ThumbDistal`];
      const pinkyProx = this.bones[`${side}LittleProximal`] || this.bones[`${side}IndexProximal`];

      if (!thumbProx || !thumbDist || !pinkyProx) continue;

      const targetPos = pinkyProx.getWorldPosition(new THREE.Vector3());

      const testAxes = [
        { name: 'y', vec: new THREE.Vector3(0, 1, 0) },
        { name: 'z', vec: new THREE.Vector3(0, 0, 1) },
        { name: 'x', vec: new THREE.Vector3(1, 0, 0) }
      ];

      // A. Calibrate ThumbProximal (opposition across palm toward pinky)
      thumbProx.quaternion.copy(thumbProx.userData.restLocalQuat);
      if (thumbInter) thumbInter.quaternion.copy(thumbInter.userData.restLocalQuat);
      thumbDist.quaternion.copy(thumbDist.userData.restLocalQuat);
      vrm.scene.updateMatrixWorld(true);

      const restDist = thumbDist.getWorldPosition(new THREE.Vector3()).distanceTo(targetPos);
      const restY = thumbDist.getWorldPosition(new THREE.Vector3()).y;

      let bestScore = -Infinity;
      let bestAxis = 'y';
      let bestSign = 1;

      for (const axis of testAxes) {
        for (const testSign of [1, -1]) {
          const qTest = new THREE.Quaternion().setFromAxisAngle(axis.vec, testSign * 0.4);
          thumbProx.quaternion.copy(thumbProx.userData.restLocalQuat).multiply(qTest);
          vrm.scene.updateMatrixWorld(true);

          const curPos = thumbDist.getWorldPosition(new THREE.Vector3());
          const curDist = curPos.distanceTo(targetPos);
          const curY = curPos.y;

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
      thumbProx.quaternion.copy(thumbProx.userData.restLocalQuat);
      vrm.scene.updateMatrixWorld(true);

      // B. Calibrate ThumbIntermediate & ThumbDistal (flexion into the palm)
      if (thumbInter) {
        let bestInterScore = -Infinity;
        let bestInterAxis = 'z';
        let bestInterSign = 1;

        thumbInter.quaternion.copy(thumbInter.userData.restLocalQuat);
        thumbDist.quaternion.copy(thumbDist.userData.restLocalQuat);
        vrm.scene.updateMatrixWorld(true);

        const restInterDist = thumbDist.getWorldPosition(new THREE.Vector3()).distanceTo(targetPos);
        const restInterY = thumbDist.getWorldPosition(new THREE.Vector3()).y;

        for (const axis of testAxes) {
          for (const testSign of [1, -1]) {
            const qTest = new THREE.Quaternion().setFromAxisAngle(axis.vec, testSign * 0.4);
            thumbInter.quaternion.copy(thumbInter.userData.restLocalQuat).multiply(qTest);
            vrm.scene.updateMatrixWorld(true);

            const curPos = thumbDist.getWorldPosition(new THREE.Vector3());
            const curDist = curPos.distanceTo(targetPos);
            const curY = curPos.y;

            const distImprovement = (restInterDist - curDist);
            const yDrop = (restInterY - curY);
            const score = distImprovement * 2.0 + yDrop;

            if (score > bestInterScore) {
              bestInterScore = score;
              bestInterAxis = axis.name;
              bestInterSign = testSign;
            }
          }
        }

        thumbInter.userData.curlAxis = bestInterAxis;
        thumbInter.userData.curlSign = bestInterSign;
        thumbInter.quaternion.copy(thumbInter.userData.restLocalQuat);

        thumbDist.userData.curlAxis = bestInterAxis;
        thumbDist.userData.curlSign = bestInterSign;
        thumbDist.quaternion.copy(thumbDist.userData.restLocalQuat);
        vrm.scene.updateMatrixWorld(true);
      } else {
        thumbDist.userData.curlAxis = bestAxis;
        thumbDist.userData.curlSign = bestSign;
      }

      // Restore rest
      thumbProx.quaternion.copy(thumbProx.userData.restLocalQuat);
      if (thumbInter) thumbInter.quaternion.copy(thumbInter.userData.restLocalQuat);
      thumbDist.quaternion.copy(thumbDist.userData.restLocalQuat);
      vrm.scene.updateMatrixWorld(true);

      console.log(`[AutoCalibrate] ${side}ThumbProx curlAxis: ${bestAxis}, sign: ${bestSign}; ThumbInter curlAxis: ${thumbInter?.userData?.curlAxis}, sign: ${thumbInter?.userData?.curlSign}`);
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

    // Max Angular Velocity Clamp: max 0.15 rad per frame (~8.6 deg at 30-60 fps)
    // Prevents sudden wild arm whipping or teleporting when landmarks flip
    const rotAngle = boneNode.quaternion.angleTo(targetLocalQuat);
    if (rotAngle < 0.012) return; // Micro-jitter deadzone eliminates high-frequency twitching

    const maxStep = 0.15;
    const step = Math.min(alpha * rotAngle, maxStep);

    boneNode.quaternion.rotateTowards(targetLocalQuat, step);
    boneNode.updateMatrixWorld(true);
  }

  /**
   * Smoothly slerp an inactive arm back to a relaxed natural posture hanging straight down at her side.
   * Uses world-direction alignment so the arm is guaranteed to hang vertically straight down.
   */
  recoverRestPose(side, deltaTime = 0.016) {
    const isLeft = side === 'left';
    const upperBone = this.bones[isLeft ? VRMHumanBoneName.LeftUpperArm : VRMHumanBoneName.RightUpperArm];
    const lowerBone = this.bones[isLeft ? VRMHumanBoneName.LeftLowerArm : VRMHumanBoneName.RightLowerArm];
    const handBone = this.bones[isLeft ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand];

    const dt = Math.max(0.001, Math.min(0.1, deltaTime));
    const alphaRest = 1.0 - Math.exp(-8.0 * dt);
    const maxStep = 0.15;

    // Target world direction: straight down along the body side with slight clearance
    const downDir = new THREE.Vector3(isLeft ? 0.06 : -0.06, -0.998, 0.02).normalize();

    if (upperBone && lowerBone) {
      this.alignBoneTowards(upperBone, lowerBone, downDir, dt);
    }

    if (lowerBone && handBone) {
      this.alignBoneTowards(lowerBone, handBone, downDir, dt);
    }

    if (handBone && handBone.userData.restLocalQuat) {
      const angle = handBone.quaternion.angleTo(handBone.userData.restLocalQuat);
      handBone.quaternion.rotateTowards(handBone.userData.restLocalQuat, Math.min(alphaRest * angle, maxStep));
      handBone.updateMatrixWorld(true);
    }

    this.relaxFingers(side, dt);
  }

  relaxFingers(side, dt = 0.016) {
    const isLeft = side === 'left';
    const prefix = isLeft ? 'left' : 'right';
    const fingerNames = ['Index', 'Middle', 'Ring', 'Little'];
    const alphaRelax = 1.0 - Math.exp(-8.0 * dt);

    for (const f of fingerNames) {
      for (const seg of ['Proximal', 'Intermediate', 'Distal']) {
        const boneKey = `${prefix}${f}${seg}`;
        const node = this.bones[boneKey];
        if (node && node.userData.restLocalQuat) {
          node.userData.smoothedCurl = 0;
          const signZ = node.userData.curlSignZ !== undefined ? node.userData.curlSignZ : (isLeft ? 1 : -1);
          // 5 degrees micro-curve inward so fingers never hyperextend backward
          const relaxedCurve = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), signZ * 0.08);
          const target = node.userData.restLocalQuat.clone().multiply(relaxedCurve);
          node.quaternion.slerp(target, alphaRelax);
          node.updateMatrixWorld(true);
        }
      }
    }

    // Natural rest posture for thumb (soft inward posture toward palm, not sticking out)
    const thumbProx = this.bones[`${prefix}ThumbProximal`];
    const thumbInter = this.bones[`${prefix}ThumbIntermediate`];
    const thumbDist = this.bones[`${prefix}ThumbDistal`];

    if (thumbProx && thumbProx.userData.restLocalQuat) {
      thumbProx.userData.smoothedFold = 0.6;
      thumbProx.userData.smoothedOpen = 0.4;
      const axisName = thumbProx.userData.curlAxis || 'y';
      const sign = thumbProx.userData.curlSign !== undefined ? thumbProx.userData.curlSign : (isLeft ? -1 : 1);
      const axisVec = axisName === 'z' ? new THREE.Vector3(0, 0, 1) :
                      axisName === 'x' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
      const qRelaxProx = new THREE.Quaternion().setFromAxisAngle(axisVec, sign * 0.35);
      thumbProx.quaternion.slerp(thumbProx.userData.restLocalQuat.clone().multiply(qRelaxProx), alphaRelax);
      thumbProx.updateMatrixWorld(true);
    }

    if (thumbInter && thumbInter.userData.restLocalQuat) {
      const axisName = thumbInter.userData.curlAxis || 'z';
      const sign = thumbInter.userData.curlSign !== undefined ? thumbInter.userData.curlSign : (isLeft ? 1 : -1);
      const axisVec = axisName === 'z' ? new THREE.Vector3(0, 0, 1) :
                      axisName === 'x' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
      const qRelaxInter = new THREE.Quaternion().setFromAxisAngle(axisVec, sign * 0.45);
      thumbInter.quaternion.slerp(thumbInter.userData.restLocalQuat.clone().multiply(qRelaxInter), alphaRelax);
      thumbInter.updateMatrixWorld(true);
    }

    if (thumbDist && thumbDist.userData.restLocalQuat) {
      const axisName = thumbInter?.userData?.curlAxis || 'z';
      const sign = thumbInter?.userData?.curlSign !== undefined ? thumbInter.userData.curlSign : (isLeft ? 1 : -1);
      const axisVec = axisName === 'z' ? new THREE.Vector3(0, 0, 1) :
                      axisName === 'x' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
      const qRelaxDist = new THREE.Quaternion().setFromAxisAngle(axisVec, sign * 0.40);
      thumbDist.quaternion.slerp(thumbDist.userData.restLocalQuat.clone().multiply(qRelaxDist), alphaRelax);
      thumbDist.updateMatrixWorld(true);
    }
  }

  /**
   * Anatomical Wrist & Finger Solver
   * 1. 2D Knuckle Chirality detects Palmar (+Z) vs Dorsal (-Z) hand facing.
   * 2. Full 3D Orthonormal Basis rotates Wrist so back of hand or palm faces camera accurately.
   * 3. 100% Cyan-Skeleton matching finger solver with unidirectional anatomical clamp (0° to 105° inward).
   */
  solveHandAndFingers(side, skelHandWorld, deltaTime = 0.016) {
    if (!skelHandWorld || skelHandWorld.length < 21) {
      this.recoverRestPose(side, deltaTime);
      return;
    }

    const dt = Math.max(0.001, Math.min(0.1, deltaTime));
    const isLeft = side === 'left';
    const prefix = isLeft ? 'left' : 'right';
    const sideSign = isLeft ? 1 : -1;

    const p0 = skelHandWorld[0];
    const p9 = skelHandWorld[9];
    const p5 = skelHandWorld[5];
    const p17 = skelHandWorld[17];

    // 1. Exact Dorsal (Mu bàn tay) vs Palmar (Lòng bàn tay) Detection via 2D Knuckle Chirality
    const palmVecX = p9.x - p0.x;
    const palmVecY = p9.y - p0.y;
    const knuckleVecX = p5.x - p17.x;
    const knuckleVecY = p5.y - p17.y;
    const palmWidth = Math.hypot(palmVecX, palmVecY) + 1e-5;

    const chirality = (palmVecX * knuckleVecY - palmVecY * knuckleVecX) / (palmWidth * palmWidth);
    const rawFacing = THREE.MathUtils.clamp(chirality * sideSign * 2.8, -1.0, 1.0);

    const facingKey = `${prefix}SmoothedFacing`;
    if (this[facingKey] === undefined) {
      this[facingKey] = rawFacing;
    } else {
      this[facingKey] += (rawFacing - this[facingKey]) * (1.0 - Math.exp(-10.0 * dt));
    }
    const smoothedFacing = this[facingKey];
    const sidewaysFactor = Math.sqrt(Math.max(0.0, 1.0 - smoothedFacing * smoothedFacing));

    // 2. Primary Hand Axis & 3D Palm Normal Vector
    const axisX = p9.clone().sub(p0).normalize();

    // Target normal: points toward camera (+Z) when palm faces out, toward chest (-Z) when back of hand faces out,
    // and toward body midline (-sideSign on X) when sideways
    const midlineX = -sideSign;
    const targetNormal = new THREE.Vector3(midlineX * sidewaysFactor, 0, smoothedFacing).normalize();
    let palmNormal = targetNormal.clone().sub(axisX.clone().multiplyScalar(targetNormal.dot(axisX))).normalize();
    if (palmNormal.lengthSq() < 1e-4) {
      const fallbackUp = new THREE.Vector3(0, 1, 0);
      palmNormal = fallbackUp.clone().sub(axisX.clone().multiplyScalar(fallbackUp.dot(axisX))).normalize();
    }

    // 3. Exact 3D Orthonormal Basis Alignment for Wrist (LeftHand / RightHand)
    const rawPalmLen = Math.max(p9.distanceTo(p0), 0.05);
    const dIdxFast = skelHandWorld[8].distanceTo(skelHandWorld[5]);
    const dMidFast = skelHandWorld[12].distanceTo(skelHandWorld[9]);
    const dRngFast = skelHandWorld[16].distanceTo(skelHandWorld[13]);
    const isFistHandFast = (dIdxFast < 0.38 * rawPalmLen && dMidFast < 0.38 * rawPalmLen && dRngFast < 0.38 * rawPalmLen) ||
                           (this[`${prefix}IsFist`] === true && dIdxFast < 0.42 * rawPalmLen && dMidFast < 0.42 * rawPalmLen && dRngFast < 0.42 * rawPalmLen);

    const handNode = this.bones[isLeft ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand];
    if (handNode && handNode.userData.restLocalQuat) {
      const v1_rest = (handNode.userData.localRestDir || new THREE.Vector3(isLeft ? 1 : -1, 0, 0)).clone().normalize();
      let v2_rest = new THREE.Vector3(0, -1, 0); // VRM T-pose inner palm points -Y
      v2_rest = v2_rest.sub(v1_rest.clone().multiplyScalar(v2_rest.dot(v1_rest))).normalize();
      const v3_rest = new THREE.Vector3().crossVectors(v1_rest, v2_rest).normalize();

      const v1_target = axisX.clone();
      const v2_target = palmNormal.clone();
      const v3_target = new THREE.Vector3().crossVectors(v1_target, v2_target).normalize();

      const mRest = new THREE.Matrix4().makeBasis(v1_rest, v2_rest, v3_rest);
      const mTarget = new THREE.Matrix4().makeBasis(v1_target, v2_target, v3_target);
      const mWorld = mTarget.clone().multiply(mRest.clone().transpose());
      const targetWorldQuat = new THREE.Quaternion().setFromRotationMatrix(mWorld);

      const parentWorldQuat = handNode.parent
        ? handNode.parent.getWorldQuaternion(new THREE.Quaternion())
        : new THREE.Quaternion();
      const targetLocalQuat = parentWorldQuat.clone().invert().multiply(targetWorldQuat);

      const angleWrist = handNode.quaternion.angleTo(targetLocalQuat);
      const alphaWrist = isFistHandFast ? (1.0 - Math.exp(-6.0 * dt)) : (1.0 - Math.exp(-12.0 * dt));
      const maxWristStep = isFistHandFast ? 0.08 : 0.18;

      // Micro-jitter deadzone for clenched fist: ignore small landmark twitching (< 2 deg)
      if (isFistHandFast && angleWrist < 0.035) {
        // Keep wrist completely steady
      } else {
        handNode.quaternion.rotateTowards(targetLocalQuat, Math.min(alphaWrist * angleWrist, maxWristStep));
        handNode.updateMatrixWorld(true);
      }
    }

    // 4. Smooth palm length denominator across frames (smoothedPalmLen)
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

    // 5. Multi-Metric Cyan-Skeleton Finger & Gesture Solver
    const knucklePlaneDir = new THREE.Vector3().crossVectors(axisX, palmNormal).normalize();
    const fingerAlpha = 1.0 - Math.exp(-12.0 * dt);

    const fingerDefs = [
      { name: 'Index', mcp: 5, pip: 6, dip: 7, tip: 8 },
      { name: 'Middle', mcp: 9, pip: 10, dip: 11, tip: 12 },
      { name: 'Ring', mcp: 13, pip: 14, dip: 15, tip: 16 },
      { name: 'Little', mcp: 17, pip: 18, dip: 19, tip: 20 }
    ];

    // Pass 1: Compute individual finger metrics
    const fingerMetrics = {};
    for (const f of fingerDefs) {
      const pMcp = skelHandWorld[f.mcp].clone();
      const pPip = skelHandWorld[f.pip].clone();
      const pDip = skelHandWorld[f.dip].clone();
      const pTip = skelHandWorld[f.tip].clone();

      const directDist = pTip.distanceTo(pMcp);
      const curveDist = Math.max(pPip.distanceTo(pMcp) + pDip.distanceTo(pPip) + pTip.distanceTo(pDip), 0.01);
      const extensionRatio = directDist / curveDist;

      // Multi-metric fist / curl detector:
      // (1) Distance criterion: Tip is close to MCP (< 38% of palm length)
      const isDistanceCurled = (directDist < 0.38 * palmLen);

      // (2) Angular criterion: Angle between MCP->PIP and PIP->TIP exceeds 95 deg (> 1.60 rad)
      const vMcpPip = pPip.clone().sub(pMcp).normalize();
      const vPipTip = pTip.clone().sub(pPip).normalize();
      const bendAngle = vMcpPip.angleTo(vPipTip);
      const isAngleCurled = (bendAngle > 1.60);

      // (3) Straight extension criterion:
      const isStraight = (bendAngle < 0.45) && (directDist > 0.45 * palmLen) && (extensionRatio > 0.72);

      let rawCurl = 0;
      if (isDistanceCurled || isAngleCurled) {
        rawCurl = 1.0;
      } else if (isStraight) {
        rawCurl = 0.0;
      } else {
        rawCurl = THREE.MathUtils.clamp((0.82 - extensionRatio) / (0.82 - 0.32), 0.0, 1.0);
      }

      fingerMetrics[f.name] = {
        pMcp, pPip, pDip, pTip,
        directDist, curveDist, extensionRatio,
        bendAngle, isDistanceCurled, isAngleCurled, isStraight,
        rawCurl, isCurled: (rawCurl >= 0.75 || isDistanceCurled || isAngleCurled)
      };
    }

    // Pass 2: High-Level Gesture Overrides (Pointing, Numbers, Closed Fist)
    const idxM = fingerMetrics['Index'];
    const midM = fingerMetrics['Middle'];
    const ringM = fingerMetrics['Ring'];
    const litM = fingerMetrics['Little'];

    const curledCount = (idxM.isCurled ? 1 : 0) + (midM.isCurled ? 1 : 0) + (ringM.isCurled ? 1 : 0) + (litM.isCurled ? 1 : 0);

    // Robust high-level gesture classification:
    // Case A: Number 3 / Number 8 (Index, Middle, Ring straight, Little curled)
    const isNumber3 = (idxM.isStraight || idxM.rawCurl < 0.25) && 
                      (midM.isStraight || midM.rawCurl < 0.25) && 
                      (ringM.isStraight || ringM.rawCurl < 0.25) && 
                      (!litM.isStraight);

    // Case B: Number 2 / "Chúng tôi" / Number 7 (Index + Middle straight, Ring & Little curled)
    const isNumber2 = (idxM.isStraight || idxM.rawCurl < 0.25) && 
                      (midM.isStraight || midM.rawCurl < 0.25) && 
                      (!ringM.isStraight) && (!litM.isStraight);

    // Case C: Pointing / "Bạn" / "Tôi" / Number 1 / Number 6 (Index straight, Middle, Ring, Little curled)
    const isIndexPointing = (idxM.isStraight || idxM.rawCurl < 0.25) && 
                            (!midM.isStraight) && (!ringM.isStraight) && (!litM.isStraight);

    // Case D: Number 4 (all 4 fingers straight, thumb tucked across palm)
    const pThumbTip = skelHandWorld[4];
    const pPinkyMcp = skelHandWorld[17];
    const dThumbPinky = pThumbTip.distanceTo(pPinkyMcp) / palmLen;
    const isNumber4 = (idxM.isStraight || idxM.rawCurl < 0.25) && 
                      (midM.isStraight || midM.rawCurl < 0.25) && 
                      (ringM.isStraight || ringM.rawCurl < 0.25) && 
                      (litM.isStraight || litM.rawCurl < 0.25) && 
                      (dThumbPinky < 0.55);

    if (isNumber3) {
      idxM.rawCurl = 0.0;
      midM.rawCurl = 0.0;
      ringM.rawCurl = 0.0;
      litM.rawCurl = 1.0;
      litM.isLockedCurled = true;
    } else if (isNumber2) {
      idxM.rawCurl = 0.0;
      midM.rawCurl = 0.0;
      ringM.rawCurl = 1.0;
      litM.rawCurl = 1.0;
      ringM.isLockedCurled = true;
      litM.isLockedCurled = true;
    } else if (isIndexPointing) {
      idxM.rawCurl = 0.0;
      midM.rawCurl = 1.0;
      ringM.rawCurl = 1.0;
      litM.rawCurl = 1.0;
      midM.isLockedCurled = true;
      ringM.isLockedCurled = true;
      litM.isLockedCurled = true;
    } else if (isNumber4) {
      idxM.rawCurl = 0.0;
      midM.rawCurl = 0.0;
      ringM.rawCurl = 0.0;
      litM.rawCurl = 0.0;
    } else if (isFistHandFast || (curledCount >= 4) || (curledCount >= 3 && !idxM.isStraight && idxM.rawCurl > 0.45)) {
      // Clenched fist (nắm chặt tay): snap and lock ALL fingers 100% into tight fist with zero jitter
      for (const f of fingerDefs) {
        fingerMetrics[f.name].rawCurl = 1.0;
        fingerMetrics[f.name].isLockedCurled = true;
      }
    }

    const isFist = isFistHandFast || (curledCount >= 4) || (curledCount >= 3 && !idxM.isStraight && idxM.rawCurl > 0.45);
    this[`${prefix}IsFist`] = isFist;

    // Pass 3: Apply smoothed curl & rotations to finger bones
    for (const f of fingerDefs) {
      const m = fingerMetrics[f.name];
      const proxNode = this.bones[`${prefix}${f.name}Proximal`];
      const interNode = this.bones[`${prefix}${f.name}Intermediate`];
      const distNode = this.bones[`${prefix}${f.name}Distal`];

      if (!proxNode) continue;

      // Lock curled fingers with zero jitter
      if (m.isLockedCurled) {
        proxNode.userData.smoothedCurl = 1.0;
      } else {
        if (proxNode.userData.smoothedCurl === undefined) {
          proxNode.userData.smoothedCurl = m.rawCurl;
        } else {
          const prevCurl = proxNode.userData.smoothedCurl;
          if (Math.abs(m.rawCurl - prevCurl) > 0.02) {
            proxNode.userData.smoothedCurl = prevCurl + (m.rawCurl - prevCurl) * (1.0 - Math.exp(-12.0 * dt));
          }
        }
      }
      const smoothCurl = THREE.MathUtils.clamp(proxNode.userData.smoothedCurl, 0.0, 1.0);

      const signZ = proxNode.userData.curlSignZ !== undefined ? proxNode.userData.curlSignZ : (isLeft ? 1 : -1);

      // Anatomical flexion angles:
      // Tightly clenched into palm when locked curled (nắm chặt) vs smooth curl
      const finalFlexProx = m.isLockedCurled ? 1.55 : (0.05 + smoothCurl * 1.50);
      const finalFlexInter = m.isLockedCurled ? 1.95 : (0.05 + smoothCurl * 1.90);
      const finalFlexDist = m.isLockedCurled ? 1.55 : (0.05 + smoothCurl * 1.50);

      // Knuckle plane splay: STRICTLY 0.0 when finger is curled or locked!
      // Completely eliminates lateral swinging / fluttering / đung đưa on the little finger and all curled fingers!
      let finalSplay = 0.0;
      if (!m.isLockedCurled && smoothCurl < 0.35) {
        const vProx = m.pPip.clone().sub(m.pMcp).normalize();
        const dotSplay = vProx.dot(knucklePlaneDir);
        const dotAxisProx = vProx.dot(axisX);
        const rawSplay = Math.atan2(dotSplay, Math.max(0.01, dotAxisProx));
        const splayWeight = 1.0 - smoothCurl / 0.35;
        finalSplay = splayWeight * THREE.MathUtils.clamp(rawSplay, -0.15, 0.15);
      }

      // Proximal Bone Rotation (Flexion + Splay):
      if (proxNode.userData.restLocalQuat) {
        const qFlex = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), signZ * finalFlexProx);
        const qSplay = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), finalSplay);
        const targetProx = proxNode.userData.restLocalQuat.clone().multiply(qFlex).multiply(qSplay);
        proxNode.quaternion.slerp(targetProx, fingerAlpha);
        proxNode.updateMatrixWorld(true);
      }

      // Intermediate Bone Rotation (Flexion only):
      if (interNode && interNode.userData.restLocalQuat) {
        const qFlexInter = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), signZ * finalFlexInter);
        const targetInter = interNode.userData.restLocalQuat.clone().multiply(qFlexInter);
        interNode.quaternion.slerp(targetInter, fingerAlpha);
        interNode.updateMatrixWorld(true);
      }

      // Distal Bone Rotation (Flexion only):
      if (distNode && distNode.userData.restLocalQuat) {
        const qFlexDist = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), signZ * finalFlexDist);
        const targetDist = distNode.userData.restLocalQuat.clone().multiply(qFlexDist);
        distNode.quaternion.slerp(targetDist, fingerAlpha);
        distNode.updateMatrixWorld(true);
      }
    }

    // 6. Anatomical Thumb Solver (Opposition, Flexion & Abduction)
    const pThumbMcp = skelHandWorld[2];
    const pIndexMcp = skelHandWorld[5];
    const dThumbIndex = pThumbTip.distanceTo(pIndexMcp) / palmLen;

    // Detection of Thumb Extension Exceptions:
    // User rule: In most words the thumb is curled/folded (nắm), EXCEPT 5 to 9 and a few specific words (tốt, yêu, etc.)
    const curWord = (this.currentWordName || '').toLowerCase();

    // 0. Number 5: 4 fingers curled into fist, thumb extended outward
    const isNum5 = (curWord === '5') || 
                   ((curledCount >= 3) && (dThumbPinky > 0.85) && (dThumbIndex > 0.45));

    // 1. Number 6: Index straight, Middle/Ring/Little curled, Thumb extended
    const isNum6 = (curWord === '6' && (idxM.isStraight || idxM.rawCurl < 0.40)) || 
                   ((idxM.isStraight || idxM.rawCurl < 0.35) && 
                    (midM.rawCurl > 0.50 || midM.isCurled) && 
                    (ringM.rawCurl > 0.50 || ringM.isCurled) && 
                    (litM.rawCurl > 0.50 || litM.isCurled) && 
                    (dThumbIndex > 0.38) && (dThumbPinky > 0.70));

    // 2. Number 7: Index + Middle straight, Ring + Little curled, Thumb extended
    const isNum7 = (curWord === '7' && (idxM.isStraight || idxM.rawCurl < 0.40)) || 
                   ((idxM.isStraight || idxM.rawCurl < 0.35) && 
                    (midM.isStraight || midM.rawCurl < 0.35) && 
                    (ringM.rawCurl > 0.50 || ringM.isCurled) && 
                    (litM.rawCurl > 0.50 || litM.isCurled) && 
                    (dThumbIndex > 0.38) && (dThumbPinky > 0.70));

    // 3. Number 8: Index + Middle + Ring straight, Little curled, Thumb extended
    const isNum8 = (curWord === '8' && (idxM.isStraight || idxM.rawCurl < 0.40)) || 
                   ((idxM.isStraight || idxM.rawCurl < 0.35) && 
                    (midM.isStraight || midM.rawCurl < 0.35) && 
                    (ringM.isStraight || ringM.rawCurl < 0.35) && 
                    (litM.rawCurl > 0.50 || litM.isCurled) && 
                    (dThumbIndex > 0.35) && (dThumbPinky > 0.65));

    // 4. Number 9 / 10 / all-straight open hand with wide thumb: 4 fingers straight AND thumb spread far
    const isAllOpen = (idxM.isStraight || idxM.rawCurl < 0.30) && 
                      (midM.isStraight || midM.rawCurl < 0.30) && 
                      (ringM.isStraight || ringM.rawCurl < 0.30) && 
                      (litM.isStraight || litM.rawCurl < 0.30) && 
                      (dThumbPinky > 0.70) && (dThumbIndex > 0.40);

    // 5. Thumbs Up ("Tốt"): 4 fingers curled into a fist, thumb pointing UP with high distance to pinky
    const isThumbsUp = (curWord === 'tốt') || 
                       ((curledCount >= 3) && (dThumbPinky > 1.25) && (pThumbTip.y > pThumbMcp.y));

    // 6. ILY sign ("Yêu"): Index straight, Little straight, Middle & Ring curled, Thumb abducted
    const isILY = (curWord === 'yêu') || 
                  ((idxM.isStraight || idxM.rawCurl < 0.35) && (litM.isStraight || litM.rawCurl < 0.35) && 
                   (midM.rawCurl > 0.50) && (ringM.rawCurl > 0.50) && (dThumbPinky > 1.10));

    // Words where thumb is explicitly FOLDED into palm:
    const foldThumbWords = ['1', '2', '3', '4', 'tôi', 'bạn', 'chúng tôi', 'bọn họ', 'làm ơn', 'xin lỗi'];
    const isFoldWord = foldThumbWords.includes(curWord);

    // Special safeguard: on clenched fist hand (nắm chặt tay) in two-hand numbers, the fist hand always keeps thumb folded
    const isThisHandFist = isFist && !isNum5;
    const isThumbExtended = (!isThisHandFist) && (!isFoldWord) && (isNum5 || isNum6 || isNum7 || isNum8 || isAllOpen || isThumbsUp || isILY);

    // Default: thumb is FOLDED / NẮM for almost all words (1, 2, 3, 4, Bạn, Tôi, fists, etc.)
    let rawThumbFold = 1.0;
    let rawThumbOpen = 0.0;

    if (isThumbExtended) {
      rawThumbFold = 0.0;
      rawThumbOpen = 1.0;
    }

    const thumbProx = this.bones[`${prefix}ThumbProximal`] || this.bones[`${prefix}ThumbMetacarpal`];
    const thumbInter = this.bones[`${prefix}ThumbIntermediate`];
    const thumbDistNode = this.bones[`${prefix}ThumbDistal`];

    if (thumbProx && thumbProx.userData.restLocalQuat) {
      if (thumbProx.userData.smoothedFold === undefined) {
        thumbProx.userData.smoothedFold = rawThumbFold;
        thumbProx.userData.smoothedOpen = rawThumbOpen;
      } else {
        const prevFold = thumbProx.userData.smoothedFold;
        if (Math.abs(rawThumbFold - prevFold) > 0.02) {
          thumbProx.userData.smoothedFold = prevFold + (rawThumbFold - prevFold) * (1.0 - Math.exp(-12.0 * dt));
        }

        const prevOpen = thumbProx.userData.smoothedOpen;
        if (Math.abs(rawThumbOpen - prevOpen) > 0.02) {
          thumbProx.userData.smoothedOpen = prevOpen + (rawThumbOpen - prevOpen) * (1.0 - Math.exp(-12.0 * dt));
        }
      }
      const sFold = THREE.MathUtils.clamp(thumbProx.userData.smoothedFold, 0.0, 1.0);
      const sOpen = THREE.MathUtils.clamp(thumbProx.userData.smoothedOpen, 0.0, 1.0);

      const axisName = thumbProx.userData.curlAxis || 'y';
      const sign = thumbProx.userData.curlSign !== undefined ? thumbProx.userData.curlSign : (isLeft ? -1 : 1);
      const axisVec = axisName === 'z' ? new THREE.Vector3(0, 0, 1) :
                      axisName === 'x' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);

      // Inward opposition across palm (sFold) vs Outward abduction away from palm (sOpen)
      const foldScaleProx = isFist ? 1.15 : 0.85;
      const foldScaleInter = isFist ? 1.35 : 1.00;
      const foldScaleDist = isFist ? 1.25 : 0.95;

      const qProx = new THREE.Quaternion()
        .setFromAxisAngle(axisVec, sign * (sFold * foldScaleProx - sOpen * 0.55))
        .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), sign * (sFold * 0.35 - sOpen * 0.20)));

      const targetProx = thumbProx.userData.restLocalQuat.clone().multiply(qProx);
      thumbProx.quaternion.slerp(targetProx, fingerAlpha);
      thumbProx.updateMatrixWorld(true);

      // Intermediate thumb bone (Thumb2 MCP flexion into the palm when folded, straight when open)
      if (thumbInter && thumbInter.userData.restLocalQuat) {
        const interAxisName = thumbInter.userData.curlAxis || 'z';
        const interSign = thumbInter.userData.curlSign !== undefined ? thumbInter.userData.curlSign : sign;
        const interAxisVec = interAxisName === 'z' ? new THREE.Vector3(0, 0, 1) :
                             interAxisName === 'x' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);

        const qFoldInter = new THREE.Quaternion().setFromAxisAngle(interAxisVec, interSign * (sFold * foldScaleInter));
        const targetInter = thumbInter.userData.restLocalQuat.clone().multiply(qFoldInter);
        thumbInter.quaternion.slerp(targetInter, fingerAlpha);
        thumbInter.updateMatrixWorld(true);
      }

      // Distal thumb bone (Thumb3 IP flexion / clamping when folded, straight when open)
      if (thumbDistNode && thumbDistNode.userData.restLocalQuat) {
        const distAxisName = thumbInter?.userData?.curlAxis || thumbDistNode.userData.curlAxis || 'z';
        const distSign = thumbInter?.userData?.curlSign !== undefined ? thumbInter.userData.curlSign : sign;
        const distAxisVec = distAxisName === 'z' ? new THREE.Vector3(0, 0, 1) :
                            distAxisName === 'x' ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);

        const qTip = new THREE.Quaternion().setFromAxisAngle(distAxisVec, distSign * (sFold * (foldScaleDist + 0.05)));
        const targetDist = thumbDistNode.userData.restLocalQuat.clone().multiply(qTip);
        thumbDistNode.quaternion.slerp(targetDist, fingerAlpha);
        thumbDistNode.updateMatrixWorld(true);
      }
    }
  }

  /**
   * Main Solver Entry Point: Read 3D skeleton joints in READ-ONLY mode.
   */
  update(skeletonWorldData, delta = 0.016) {
    if (!this.isLoaded || !this.currentVRM || !skeletonWorldData) return;

    this.currentWordName = (skeletonWorldData.wordName || '').trim().toLowerCase();

    const { poseWorld, leftHandWorld, rightHandWorld } = skeletonWorldData;
    if (!poseWorld || poseWorld.length < 8) return;

    // Outlier jump clamp: prevents any single joint from jumping > 0.35m in a single frame
    if (!this.prevPoseWorld) {
      this.prevPoseWorld = poseWorld.map(p => (p ? p.clone() : null));
    } else {
      for (let j = 0; j < 8; j++) {
        if (poseWorld[j] && this.prevPoseWorld[j]) {
          const jumpDist = poseWorld[j].distanceTo(this.prevPoseWorld[j]);
          if (jumpDist > 0.35) {
            poseWorld[j].lerpVectors(this.prevPoseWorld[j], poseWorld[j], 0.35 / jumpDist);
          }
          this.prevPoseWorld[j].copy(poseWorld[j]);
        }
      }
    }

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
    // Determine active arm states for single-hand vs two-hand gestures
    const isLeftActive = skeletonWorldData.isLeftActive !== undefined
      ? skeletonWorldData.isLeftActive
      : (!!leftHandWorld && skeletonWorldData.activeHandSide !== 'rightHand');

    const isRightActive = skeletonWorldData.isRightActive !== undefined
      ? skeletonWorldData.isRightActive
      : (!!rightHandWorld && skeletonWorldData.activeHandSide !== 'leftHand');

    // Left Arm (Joints 1: Shoulder, 3: Elbow, 5: Wrist)
    if (isLeftActive && leftHandWorld && poseWorld[1] && poseWorld[3] && poseWorld[5]) {
      const leftUpperArm = this.bones[VRMHumanBoneName.LeftUpperArm];
      const leftLowerArm = this.bones[VRMHumanBoneName.LeftLowerArm];
      const leftHandBone = this.bones[VRMHumanBoneName.LeftHand];

      const targetUpperDir = poseWorld[3].clone().sub(poseWorld[1]).normalize();
      this.alignBoneTowards(leftUpperArm, leftLowerArm, targetUpperDir, delta);

      const targetLowerDir = poseWorld[5].clone().sub(poseWorld[3]).normalize();
      this.alignBoneTowards(leftLowerArm, leftHandBone, targetLowerDir, delta);

      this.solveHandAndFingers('left', leftHandWorld, delta);
    } else {
      // Inactive or single-hand inactive arm: luôn duỗi thẳng tự nhiên xuống dưới
      this.recoverRestPose('left', delta);
    }

    // Right Arm (Joints 2: Shoulder, 4: Elbow, 6: Wrist)
    if (isRightActive && rightHandWorld && poseWorld[2] && poseWorld[4] && poseWorld[6]) {
      const rightUpperArm = this.bones[VRMHumanBoneName.RightUpperArm];
      const rightLowerArm = this.bones[VRMHumanBoneName.RightLowerArm];
      const rightHandBone = this.bones[VRMHumanBoneName.RightHand];

      const targetUpperDir = poseWorld[4].clone().sub(poseWorld[2]).normalize();
      this.alignBoneTowards(rightUpperArm, rightLowerArm, targetUpperDir, delta);

      const targetLowerDir = poseWorld[6].clone().sub(poseWorld[4]).normalize();
      this.alignBoneTowards(rightLowerArm, rightHandBone, targetLowerDir, delta);

      this.solveHandAndFingers('right', rightHandWorld, delta);
    } else {
      // Inactive or single-hand inactive arm: luôn duỗi thẳng tự nhiên xuống dưới
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
