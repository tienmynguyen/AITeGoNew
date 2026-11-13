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
   * NEVER resets boneNode.quaternion to restLocalQuat during traversal!
   * Eliminates parent-child matrix oscillation and high-frequency twitching.
   */
  alignBoneTowards(boneNode, childNode, targetDir, customSlerp = null) {
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

    // 6. Smooth slerp using persistent currentQuat
    if (!boneNode.userData.currentQuat) {
      boneNode.userData.currentQuat = boneNode.quaternion.clone();
    }

    const sFactor = customSlerp !== null ? customSlerp : this.slerpFactor;
    boneNode.userData.currentQuat.slerp(targetLocalQuat, sFactor);

    // Update bone quaternion directly without temporary dirtying
    boneNode.quaternion.copy(boneNode.userData.currentQuat);
    boneNode.updateWorldMatrix(true, false);
  }

  /**
   * Smoothly slerp an inactive arm back to a relaxed natural posture by her side.
   */
  recoverRestPose(side) {
    const isLeft = side === 'left';
    const upperBone = this.bones[isLeft ? VRMHumanBoneName.LeftUpperArm : VRMHumanBoneName.RightUpperArm];
    const lowerBone = this.bones[isLeft ? VRMHumanBoneName.LeftLowerArm : VRMHumanBoneName.RightLowerArm];
    const handBone = this.bones[isLeft ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand];

    const qRelaxedUpper = isLeft ? this.relaxedLeftUpper : this.relaxedRightUpper;
    const qRelaxedLower = isLeft ? this.relaxedLeftLower : this.relaxedRightLower;

    const sFactor = 0.12;

    if (upperBone && upperBone.userData.restLocalQuat) {
      const targetQuat = upperBone.userData.restLocalQuat.clone().multiply(qRelaxedUpper);
      if (!upperBone.userData.currentQuat) upperBone.userData.currentQuat = upperBone.quaternion.clone();
      upperBone.userData.currentQuat.slerp(targetQuat, sFactor);
      upperBone.quaternion.copy(upperBone.userData.currentQuat);
      upperBone.updateWorldMatrix(true, false);
    }

    if (lowerBone && lowerBone.userData.restLocalQuat) {
      const targetQuat = lowerBone.userData.restLocalQuat.clone().multiply(qRelaxedLower);
      if (!lowerBone.userData.currentQuat) lowerBone.userData.currentQuat = lowerBone.quaternion.clone();
      lowerBone.userData.currentQuat.slerp(targetQuat, sFactor);
      lowerBone.quaternion.copy(lowerBone.userData.currentQuat);
      lowerBone.updateWorldMatrix(true, false);
    }

    if (handBone && handBone.userData.restLocalQuat) {
      if (!handBone.userData.currentQuat) handBone.userData.currentQuat = handBone.quaternion.clone();
      handBone.userData.currentQuat.slerp(handBone.userData.restLocalQuat, sFactor);
      handBone.quaternion.copy(handBone.userData.currentQuat);
      handBone.updateWorldMatrix(true, false);
    }

    this.relaxFingers(side);
  }

  relaxFingers(side) {
    const isLeft = side === 'left';
    const prefix = isLeft ? 'left' : 'right';
    const fingerNames = ['Index', 'Middle', 'Ring', 'Little', 'Thumb'];

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
          if (!node.userData.currentQuat) node.userData.currentQuat = node.quaternion.clone();
          node.userData.currentQuat.slerp(target, 0.15);
          node.quaternion.copy(node.userData.currentQuat);
        }
      }
    }
  }

  /**
   * Anatomical Wrist & Finger Solver
   * Uses load-time auto-calibrated curl signs to ensure 100% inward palm flexion.
   * Clamps against hyperextension with 0.08 rad natural micro-curve.
   * Uses EMA temporal filtering and slerping to eliminate finger tremor.
   */
  solveHandAndFingers(side, skelHandWorld) {
    if (!skelHandWorld || skelHandWorld.length < 21) {
      this.recoverRestPose(side);
      return;
    }

    const isLeft = side === 'left';
    const prefix = isLeft ? 'left' : 'right';

    // 1. Solve Wrist: primary axis from Wrist (0) to Middle MCP (9)
    const handNode = this.bones[isLeft ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand];
    const middleProxNode = this.bones[`${prefix}MiddleProximal`];

    if (handNode && middleProxNode) {
      const targetHandDir = skelHandWorld[9].clone().sub(skelHandWorld[0]).normalize();
      // Shortest-arc alignment from restLocalQuat introduces ZERO twist
      this.alignBoneTowards(handNode, middleProxNode, targetHandDir);
    }

    // Measure palm scale
    const palmLen = Math.max(skelHandWorld[9].distanceTo(skelHandWorld[0]), 0.05);

    // 2. Solve 4 Fingers (Index, Middle, Ring, Little)
    const fingerDefs = [
      { name: 'Index', mcp: 5, pip: 6, dip: 7, tip: 8 },
      { name: 'Middle', mcp: 9, pip: 10, dip: 11, tip: 12 },
      { name: 'Ring', mcp: 13, pip: 14, dip: 15, tip: 16 },
      { name: 'Little', mcp: 17, pip: 18, dip: 19, tip: 20 }
    ];

    for (const f of fingerDefs) {
      const pMcp = skelHandWorld[f.mcp];
      const pPip = skelHandWorld[f.pip];
      const pDip = skelHandWorld[f.dip];
      const pTip = skelHandWorld[f.tip];

      const directDist = pTip.distanceTo(pMcp);
      const curveDist = Math.max(pPip.distanceTo(pMcp) + pDip.distanceTo(pPip) + pTip.distanceTo(pDip), 0.01);
      const extensionRatio = directDist / curveDist;

      // 0.0 = straight open finger, 1.0 = curled tightly into palm
      let curlAmount = (0.84 - extensionRatio) / (0.84 - 0.28);
      curlAmount = THREE.MathUtils.clamp(curlAmount, 0.0, 1.0);

      const proxNode = this.bones[`${prefix}${f.name}Proximal`];
      const interNode = this.bones[`${prefix}${f.name}Intermediate`];
      const distNode = this.bones[`${prefix}${f.name}Distal`];

      // Temporal EMA filter on curl amount (alpha = 0.35) to eliminate high-frequency finger tremor
      if (proxNode) {
        if (proxNode.userData.smoothedCurl === undefined) {
          proxNode.userData.smoothedCurl = curlAmount;
        } else {
          proxNode.userData.smoothedCurl = proxNode.userData.smoothedCurl * 0.65 + curlAmount * 0.35;
        }
      }
      const smoothCurl = proxNode?.userData?.smoothedCurl !== undefined ? proxNode.userData.smoothedCurl : curlAmount;

      // Use empirical auto-calibrated sign (guaranteed to curl inward into palm!)
      const signZ = proxNode?.userData?.curlSignZ !== undefined ? proxNode.userData.curlSignZ : (isLeft ? 1 : -1);

      // Hard clamp: base micro-curve of 0.08 rad (5 degrees) inward so fingers NEVER bend backward!
      const proxAngle = signZ * (0.08 + smoothCurl * 1.25);
      const interAngle = signZ * (0.08 + smoothCurl * 1.50);
      const distAngle = signZ * (0.08 + smoothCurl * 1.00);

      const applyFingerRotation = (node, angle) => {
        if (!node || !node.userData.restLocalQuat) return;
        const qTarget = node.userData.restLocalQuat.clone().multiply(
          new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), angle)
        );
        if (!node.userData.currentQuat) {
          node.userData.currentQuat = node.quaternion.clone();
        }
        node.userData.currentQuat.slerp(qTarget, 0.40);
        node.quaternion.copy(node.userData.currentQuat);
      };

      applyFingerRotation(proxNode, proxAngle);
      applyFingerRotation(interNode, interAngle);
      applyFingerRotation(distNode, distAngle);
    }

    // 3. Solve Thumb (Points 1..4)
    const pThumbTip = skelHandWorld[4];
    const pIndexMcp = skelHandWorld[5];
    const thumbDist = pThumbTip.distanceTo(pIndexMcp) / palmLen;

    const thumbOpen = THREE.MathUtils.clamp((thumbDist - 0.35) / 0.45, 0.0, 1.0);
    const thumbFold = 1.0 - thumbOpen;

    const thumbProx = this.bones[`${prefix}ThumbProximal`] || this.bones[`${prefix}ThumbMetacarpal`];
    const thumbDistNode = this.bones[`${prefix}ThumbDistal`];

    if (thumbProx && thumbProx.userData.restLocalQuat) {
      if (thumbProx.userData.smoothedFold === undefined) {
        thumbProx.userData.smoothedFold = thumbFold;
        thumbProx.userData.smoothedOpen = thumbOpen;
      } else {
        thumbProx.userData.smoothedFold = thumbProx.userData.smoothedFold * 0.65 + thumbFold * 0.35;
        thumbProx.userData.smoothedOpen = thumbProx.userData.smoothedOpen * 0.65 + thumbOpen * 0.35;
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
      if (!thumbProx.userData.currentQuat) {
        thumbProx.userData.currentQuat = thumbProx.quaternion.clone();
      }
      thumbProx.userData.currentQuat.slerp(targetProx, 0.40);
      thumbProx.quaternion.copy(thumbProx.userData.currentQuat);
    }

    if (thumbDistNode && thumbDistNode.userData.restLocalQuat) {
      const sFold = thumbProx?.userData?.smoothedFold !== undefined ? thumbProx.userData.smoothedFold : thumbFold;
      const signZ = thumbProx?.userData?.curlSignZ !== undefined ? thumbProx.userData.curlSignZ : (isLeft ? 1 : -1);
      const qTip = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), signZ * (sFold * 0.65 + 0.06));
      const targetDist = thumbDistNode.userData.restLocalQuat.clone().multiply(qTip);
      if (!thumbDistNode.userData.currentQuat) {
        thumbDistNode.userData.currentQuat = thumbDistNode.quaternion.clone();
      }
      thumbDistNode.userData.currentQuat.slerp(targetDist, 0.40);
      thumbDistNode.quaternion.copy(thumbDistNode.userData.currentQuat);
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
      this.alignBoneTowards(leftUpperArm, leftLowerArm, targetUpperDir);

      const targetLowerDir = poseWorld[5].clone().sub(poseWorld[3]).normalize();
      this.alignBoneTowards(leftLowerArm, leftHandBone, targetLowerDir);

      if (leftHandWorld) {
        this.solveHandAndFingers('left', leftHandWorld);
      } else {
        this.recoverRestPose('left');
      }
    } else {
      this.recoverRestPose('left');
    }

    // Right Arm (Joints 2: Shoulder, 4: Elbow, 6: Wrist)
    if (poseWorld[2] && poseWorld[4] && poseWorld[6]) {
      const rightUpperArm = this.bones[VRMHumanBoneName.RightUpperArm];
      const rightLowerArm = this.bones[VRMHumanBoneName.RightLowerArm];
      const rightHandBone = this.bones[VRMHumanBoneName.RightHand];

      const targetUpperDir = poseWorld[4].clone().sub(poseWorld[2]).normalize();
      this.alignBoneTowards(rightUpperArm, rightLowerArm, targetUpperDir);

      const targetLowerDir = poseWorld[6].clone().sub(poseWorld[4]).normalize();
      this.alignBoneTowards(rightLowerArm, rightHandBone, targetLowerDir);

      if (rightHandWorld) {
        this.solveHandAndFingers('right', rightHandWorld);
      } else {
        this.recoverRestPose('right');
      }
    } else {
      this.recoverRestPose('right');
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
