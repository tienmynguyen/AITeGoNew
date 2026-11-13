import * as THREE from 'three';

/**
 * 3D Cyan Skeleton Overlay
 * Renders untouched 3D spatial depth (X, Y, Z) directly from Landmark coordinates.
 * Completely independent of the VRM bone solver.
 */
export class SkeletonOverlay {
  constructor(scene) {
    this.scene = scene;
    this.visible = true;
    this.mirror = false;

    // Anchor: Mid-Shoulder stands neatly aligned with AvatarSample_A's upper body
    this.anchor = new THREE.Vector3(0, 1.26, 0.18);
    // Equal, uniform 3D scale across X, Y, Z without flattening Z
    this.scale = 0.30;

    this.group = new THREE.Group();
    this.group.name = 'CyanSkeletonOverlay';
    this.scene.add(this.group);

    // Cyan materials
    this.lineMaterial = new THREE.LineBasicMaterial({
      color: 0x00ffff,
      linewidth: 2,
      transparent: true,
      opacity: 0.85,
      depthTest: true
    });

    this.jointMaterial = new THREE.MeshBasicMaterial({
      color: 0x00ffff,
      transparent: true,
      opacity: 0.95,
      depthTest: true
    });

    // Pose connections (8 points: 0: Nose, 1: L_Shoulder, 2: R_Shoulder, 3: L_Elbow, 4: R_Elbow, 5: L_Wrist, 6: R_Wrist, 7: Mid_Shoulder)
    this.poseEdges = [
      [0, 7], // Nose -> Mid-Shoulder
      [7, 1], // Mid-Shoulder -> Left Shoulder
      [7, 2], // Mid-Shoulder -> Right Shoulder
      [1, 3], // Left Shoulder -> Left Elbow
      [3, 5], // Left Elbow -> Left Wrist
      [2, 4], // Right Shoulder -> Right Elbow
      [4, 6]  // Right Elbow -> Right Wrist
    ];

    // Hand connections (21 points)
    this.handEdges = [
      // Palm
      [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [17, 0],
      // Thumb
      [1, 2], [2, 3], [3, 4],
      // Index
      [5, 6], [6, 7], [7, 8],
      // Middle
      [9, 10], [10, 11], [11, 12],
      // Ring
      [13, 14], [14, 15], [15, 16],
      // Pinky
      [17, 18], [18, 19], [19, 20]
    ];

    // Pre-allocated LineSegments geometry
    // Max lines: 7 pose + 21*2 hand edges = 49 lines = 98 vertices = 294 floats
    this.maxVertices = 300;
    this.linePositions = new Float32Array(this.maxVertices * 3);
    this.lineGeometry = new THREE.BufferGeometry();
    this.lineGeometry.setAttribute('position', new THREE.BufferAttribute(this.linePositions, 3));
    this.lineSegments = new THREE.LineSegments(this.lineGeometry, this.lineMaterial);
    this.group.add(this.lineSegments);

    // InstancedMesh for joint spheres
    // Max joints: 8 pose + 42 hands = 50 joints
    this.maxJoints = 60;
    const sphereGeo = new THREE.SphereGeometry(0.007, 8, 8);
    this.jointMesh = new THREE.InstancedMesh(sphereGeo, this.jointMaterial, this.maxJoints);
    this.jointMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.group.add(this.jointMesh);

    this.dummyTransform = new THREE.Object3D();
  }

  setVisible(visible) {
    this.visible = visible;
    this.group.visible = visible;
  }

  setMirror(mirror) {
    this.mirror = mirror;
  }

  /**
   * Convert normalized MediaPipe coordinates to 3D world space.
   * Preserves full, untouched 3D depth X, Y, Z.
   */
  toWorldPose(pt) {
    const signX = this.mirror ? -1 : 1;
    // In MediaPipe:
    // px: positive viewer-right
    // py: negative is upward
    // pz: negative is forward towards camera
    return new THREE.Vector3(
      this.anchor.x + signX * pt[0] * this.scale,
      this.anchor.y - pt[1] * this.scale,
      this.anchor.z - pt[2] * this.scale
    );
  }

  /**
   * Update the skeleton visualization for the current frame.
   * @param {Object} frame - { pose: [[x,y,z]x8], leftHand: [[x,y,z]x21]|null, rightHand: [[x,y,z]x21]|null }
   */
  update(frame) {
    if (!this.visible || !frame) {
      this.lineSegments.geometry.setDrawRange(0, 0);
      this.jointMesh.count = 0;
      return;
    }

    const { pose, leftHand, rightHand } = frame;
    let vertexCount = 0;
    let jointCount = 0;

    // Helper to add a 3D line segment
    const addLine = (pA, pB) => {
      const idx = vertexCount * 3;
      this.linePositions[idx] = pA.x;
      this.linePositions[idx + 1] = pA.y;
      this.linePositions[idx + 2] = pA.z;

      this.linePositions[idx + 3] = pB.x;
      this.linePositions[idx + 4] = pB.y;
      this.linePositions[idx + 5] = pB.z;
      vertexCount += 2;
    };

    // Helper to add a joint sphere
    const addJoint = (p, scale = 1.0) => {
      if (jointCount >= this.maxJoints) return;
      this.dummyTransform.position.copy(p);
      this.dummyTransform.scale.setScalar(scale);
      this.dummyTransform.updateMatrix();
      this.jointMesh.setMatrixAt(jointCount++, this.dummyTransform.matrix);
    };

    // 1. Calculate Pose world positions
    const poseWorld = [];
    if (pose && pose.length >= 8) {
      for (let i = 0; i < 8; i++) {
        const pt = pose[i];
        // Check if active (avoid drawing to floor if all 0)
        const isNonZero = (i === 7) || Math.abs(pt[0]) > 1e-4 || Math.abs(pt[1]) > 1e-4 || Math.abs(pt[2]) > 1e-4;
        if (isNonZero) {
          const w = this.toWorldPose(pt);
          poseWorld[i] = w;
          addJoint(w, i === 7 ? 1.4 : 1.0);
        } else {
          poseWorld[i] = null;
        }
      }

      // Draw Pose lines
      for (const [idxA, idxB] of this.poseEdges) {
        if (poseWorld[idxA] && poseWorld[idxB]) {
          addLine(poseWorld[idxA], poseWorld[idxB]);
        }
      }
    }

    // 2. Helper to process and render a 21-point hand skeleton
    const renderHand = (handPoints, wristAnchorWorld) => {
      if (!handPoints || handPoints.length < 21 || !wristAnchorWorld) return;

      const signX = this.mirror ? -1 : 1;
      const handWristRaw = handPoints[0];
      const handWorld = [];

      // Anchor hand[0] cleanly to wristAnchorWorld while preserving 100% relative 3D offsets
      for (let i = 0; i < 21; i++) {
        const pt = handPoints[i];
        const relX = pt[0] - handWristRaw[0];
        const relY = pt[1] - handWristRaw[1];
        const relZ = pt[2] - handWristRaw[2];

        const w = new THREE.Vector3(
          wristAnchorWorld.x + signX * relX * this.scale,
          wristAnchorWorld.y - relY * this.scale,
          wristAnchorWorld.z - relZ * this.scale
        );
        handWorld[i] = w;
        addJoint(w, i === 0 ? 1.1 : 0.7);
      }

      // Draw Hand lines
      for (const [idxA, idxB] of this.handEdges) {
        if (handWorld[idxA] && handWorld[idxB]) {
          addLine(handWorld[idxA], handWorld[idxB]);
        }
      }

      return handWorld;
    };

    // Render Left Hand (anchored to pose Left Wrist #5)
    let leftHandWorld = null;
    if (leftHand && poseWorld[5]) {
      leftHandWorld = renderHand(leftHand, poseWorld[5]);
    }

    // Render Right Hand (anchored to pose Right Wrist #6)
    let rightHandWorld = null;
    if (rightHand && poseWorld[6]) {
      rightHandWorld = renderHand(rightHand, poseWorld[6]);
    }

    // Update GPU buffers
    this.lineGeometry.attributes.position.needsUpdate = true;
    this.lineGeometry.setDrawRange(0, vertexCount);
    this.jointMesh.count = jointCount;
    this.jointMesh.instanceMatrix.needsUpdate = true;

    // Return the untouched 3D world coordinates for read-only consumption by VRM solver
    return {
      poseWorld,
      leftHandWorld,
      rightHandWorld
    };
  }
}

