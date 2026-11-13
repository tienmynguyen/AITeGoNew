import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, VRMHumanBoneName } from '@pixiv/three-vrm';

/**
 * VRM 0.0 Self-Calibrating Engine
 * Reads the 3D Cyan Skeleton positions in READ-ONLY mode.
 * Performs hierarchical arm alignment, anatomical wrist roll clamping,
 * and inward finger curling without flipping or backward hyperextension.
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

          // Capture True T-Pose Rest Data on VRM Load
          this.initRestData();

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
        node.userData.prevQuat = node.quaternion.clone();
        this.bones[boneName] = node;
      }
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
   * Solve a bone to aim towards targetDirection in world space.
   */
  alignBoneTowards(boneNode, childNode, targetDir, customSlerp = null) {
    if (!boneNode || !childNode || !targetDir || targetDir.lengthSq() < 1e-4) return;

    // 1. Reset bone to rest orientation
    boneNode.quaternion.copy(boneNode.userData.restLocalQuat);
    boneNode.updateWorldMatrix(true, false);

    // 2. Measure current rest direction in world space
    const pB = boneNode.getWorldPosition(new THREE.Vector3());
    const pC = childNode.getWorldPosition(new THREE.Vector3());
    const currentDir = pC.clone().sub(pB).normalize();

    if (currentDir.lengthSq() < 1e-4) return;

    // 3. Compute delta rotation in world space
    const deltaWorldQuat = new THREE.Quaternion().setFromUnitVectors(currentDir, targetDir);

    // 4. Convert delta rotation into bone's local space
    const worldQuatRest = boneNode.getWorldQuaternion(new THREE.Quaternion());
    const targetWorldQuat = deltaWorldQuat.multiply(worldQuatRest);

    const parentWorldQuat = boneNode.parent
      ? boneNode.parent.getWorldQuaternion(new THREE.Quaternion())
      : new THREE.Quaternion();
    const targetLocalQuat = parentWorldQuat.clone().invert().multiply(targetWorldQuat);

    // 5. Slerp from previous frame's quaternion
    const sFactor = customSlerp !== null ? customSlerp : this.slerpFactor;
    const finalQuat = boneNode.userData.prevQuat
      ? boneNode.userData.prevQuat.clone().slerp(targetLocalQuat, sFactor)
      : targetLocalQuat;

    boneNode.quaternion.copy(finalQuat);
    boneNode.userData.prevQuat.copy(finalQuat);
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
      upperBone.quaternion.slerp(targetQuat, sFactor);
      upperBone.userData.prevQuat.copy(upperBone.quaternion);
    }

    if (lowerBone && lowerBone.userData.restLocalQuat) {
      const targetQuat = lowerBone.userData.restLocalQuat.clone().multiply(qRelaxedLower);
      lowerBone.quaternion.slerp(targetQuat, sFactor);
      lowerBone.userData.prevQuat.copy(lowerBone.quaternion);
    }

    if (handBone && handBone.userData.restLocalQuat) {
      handBone.quaternion.slerp(handBone.userData.restLocalQuat, sFactor);
      handBone.userData.prevQuat.copy(handBone.quaternion);
    }

    // Reset fingers to open relaxed pose
    this.relaxFingers(side);
  }

  relaxFingers(side) {
    const isLeft = side === 'left';
    const prefix = isLeft ? 'left' : 'right';
    const curlSign = isLeft ? 1 : -1;
    const fingerNames = ['Index', 'Middle', 'Ring', 'Little', 'Thumb'];

    // 5 degrees micro-curve inward so fingers never hyperextend backward
    const relaxedCurve = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), curlSign * 0.08);

    for (const f of fingerNames) {
      for (const seg of ['Proximal', 'Intermediate', 'Distal']) {
        const boneKey = `${prefix}${f}${seg}`;
        const node = this.bones[boneKey];
        if (node && node.userData.restLocalQuat) {
          const target = node.userData.restLocalQuat.clone().multiply(relaxedCurve);
          node.quaternion.slerp(target, 0.2);
        }
      }
    }
  }

  /**
   * Anatomical Wrist & Finger Solver (Fix 3 & Fix 4)
   * Inverts curl sign so fingers curl INWARD toward the palm, never backward!
   * Clamps against hyperextension with 0.08 rad natural micro-curve.
   */
  solveHandAndFingers(side, skelHandWorld) {
    if (!skelHandWorld || skelHandWorld.length < 21) {
      this.recoverRestPose(side);
      return;
    }

    const isLeft = side === 'left';
    const prefix = isLeft ? 'left' : 'right';

    // Fix 3: Invert finger curl sign!
    // For Left Hand: POSITIVE Z rotation (+curlAngle) curls inward toward palm (-Y).
    // For Right Hand: NEGATIVE Z rotation (-curlAngle) curls inward toward palm (-Y).
    const curlSign = isLeft ? 1 : -1;

    // 1. Solve Wrist: primary axis from Wrist (0) to Middle MCP (9)
    const handNode = this.bones[isLeft ? VRMHumanBoneName.LeftHand : VRMHumanBoneName.RightHand];
    const middleProxNode = this.bones[`${prefix}MiddleProximal`];

    if (handNode && middleProxNode) {
      const targetHandDir = skelHandWorld[9].clone().sub(skelHandWorld[0]).normalize();
      this.alignBoneTowards(handNode, middleProxNode, targetHandDir);

      // Orient palm normal so inner palm faces forward (+Z) or toward midline, NEVER backward
      const vAcross = skelHandWorld[17].clone().sub(skelHandWorld[5]).normalize();
      let palmNormal = targetHandDir.clone().cross(vAcross).normalize();

      // Gentle roll clamping without sign flips
      const rollAngle = THREE.MathUtils.clamp(palmNormal.y * 0.85, -0.95, 0.95);
      const qRoll = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), rollAngle * 0.45);
      handNode.quaternion.multiply(qRoll);
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

      // Fix 3: Hard clamp against backward hyperextension!
      // Base micro-curve of 0.08 rad (5 degrees) inward toward palm so fingers never bend backward!
      const proxAngle = curlSign * (0.08 + curlAmount * 1.25);
      const interAngle = curlSign * (0.08 + curlAmount * 1.50);
      const distAngle = curlSign * (0.08 + curlAmount * 1.00);

      const proxNode = this.bones[`${prefix}${f.name}Proximal`];
      const interNode = this.bones[`${prefix}${f.name}Intermediate`];
      const distNode = this.bones[`${prefix}${f.name}Distal`];

      if (proxNode && proxNode.userData.restLocalQuat) {
        const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), proxAngle);
        proxNode.quaternion.copy(proxNode.userData.restLocalQuat).multiply(q);
      }
      if (interNode && interNode.userData.restLocalQuat) {
        const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), interAngle);
        interNode.quaternion.copy(interNode.userData.restLocalQuat).multiply(q);
      }
      if (distNode && distNode.userData.restLocalQuat) {
        const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), distAngle);
        distNode.quaternion.copy(distNode.userData.restLocalQuat).multiply(q);
      }
    }

    // 3. Solve Thumb (Points 1..4)
    const pThumbTip = skelHandWorld[4];
    const pIndexMcp = skelHandWorld[5];
    const thumbDist = pThumbTip.distanceTo(pIndexMcp) / palmLen;

    const thumbOpen = THREE.MathUtils.clamp((thumbDist - 0.35) / 0.45, 0.0, 1.0);
    const thumbFold = 1.0 - thumbOpen;

    const thumbProx = this.bones[`${prefix}ThumbProximal`];
    const thumbDistNode = this.bones[`${prefix}ThumbDistal`];

    if (thumbProx && thumbProx.userData.restLocalQuat) {
      const qFold = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), curlSign * thumbFold * 0.55);
      const qAbduct = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), curlSign * (thumbOpen * 0.35 - thumbFold * 0.35));
      thumbProx.quaternion.copy(thumbProx.userData.restLocalQuat).multiply(qFold).multiply(qAbduct);
    }

    if (thumbDistNode && thumbDistNode.userData.restLocalQuat) {
      const qTip = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), curlSign * (thumbFold * 0.65 + 0.06));
      thumbDistNode.quaternion.copy(thumbDistNode.userData.restLocalQuat).multiply(qTip);
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
      headNode.quaternion.copy(headNode.userData.restLocalQuat).multiply(qHead);
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
