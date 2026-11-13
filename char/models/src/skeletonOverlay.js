import * as THREE from 'three';

/**
 * 3D Cyan Skeleton Overlay
 * Renders untouched volumetric 3D spatial depth (X, Y, Z).
 * Anchors directly inside AvatarSample_A's actual shoulder midpoint with zero Z separation.
 */
export class SkeletonOverlay {
  constructor(scene) {
    this.scene = scene;
    this.visible = true;
    this.mirror = false;

    // Anchor initialized to human upper body, calibrated dynamically from VRM shoulders
    this.anchor = new THREE.Vector3(0, 1.18, 0.0);
    this.scale = 0.32;
    this.vrmLeftShoulder = null;
    this.vrmRightShoulder = null;

    this.group = new THREE.Group();
    this.group.name = 'CyanSkeletonOverlay';
    this.scene.add(this.group);

    // Cyan materials
    this.lineMaterial = new THREE.LineBasicMaterial({
      color: 0x00ffff,
      linewidth: 2,
      transparent: true,
      opacity: 0.90,
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
    this.maxVertices = 300;
    this.linePositions = new Float32Array(this.maxVertices * 3);
    this.lineGeometry = new THREE.BufferGeometry();
    this.lineGeometry.setAttribute('position', new THREE.BufferAttribute(this.linePositions, 3));
    this.lineSegments = new THREE.LineSegments(this.lineGeometry, this.lineMaterial);
    this.group.add(this.lineSegments);

    // InstancedMesh for joint spheres
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
   * Fix 1: Anchor directly to AvatarSample_A's actual shoulder midpoint in 3D space.
   * Zero separation in Z!
   */
  setShoulderCalibration(leftShoulderPos, rightShoulderPos) {
    if (!leftShoulderPos || !rightShoulderPos) return;

    this.vrmLeftShoulder = leftShoulderPos.clone();
    this.vrmRightShoulder = rightShoulderPos.clone();

    // Mid-shoulder midpoint
    this.anchor = leftShoulderPos.clone().add(rightShoulderPos).multiplyScalar(0.5);
    // Shoulder width as scale
    const dist = leftShoulderPos.distanceTo(rightShoulderPos);
    if (dist > 0.1) {
      this.scale = dist;
    }
  }

  /**
   * Update the skeleton visualization for the current frame.
   * Computes true 3D forward kinematic depth (Fix 2 & Fix 1).
   * @param {Object} frame - { pose: [[x,y,z]x8], leftHand, rightHand }
   */
  update(frame) {
    if (!this.visible || !frame) {
      this.lineSegments.geometry.setDrawRange(0, 0);
      this.jointMesh.count = 0;
      return null;
    }

    const { pose, leftHand, rightHand } = frame;
    let vertexCount = 0;
    let jointCount = 0;

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

    const addJoint = (p, scale = 1.0) => {
      if (jointCount >= this.maxJoints) return;
      this.dummyTransform.position.copy(p);
      this.dummyTransform.scale.setScalar(scale);
      this.dummyTransform.updateMatrix();
      this.jointMesh.setMatrixAt(jointCount++, this.dummyTransform.matrix);
    };

    const signX = this.mirror ? -1 : 1;
    const poseWorld = [];

    if (pose && pose.length >= 8) {
      // 1. Mid-Shoulder (Point 7)
      const midShoulder = this.anchor.clone();
      poseWorld[7] = midShoulder;
      addJoint(midShoulder, 1.4);

      // 2. Left Shoulder (Point 1) & Right Shoulder (Point 2)
      // Sits 100% on top of AvatarSample_A's actual shoulders
      if (this.vrmLeftShoulder && this.vrmRightShoulder) {
        poseWorld[1] = this.mirror ? this.vrmRightShoulder.clone() : this.vrmLeftShoulder.clone();
        poseWorld[2] = this.mirror ? this.vrmLeftShoulder.clone() : this.vrmRightShoulder.clone();
      } else {
        poseWorld[1] = new THREE.Vector3(
          this.anchor.x + signX * pose[1][0] * this.scale,
          this.anchor.y - pose[1][1] * this.scale,
          this.anchor.z
        );
        poseWorld[2] = new THREE.Vector3(
          this.anchor.x + signX * pose[2][0] * this.scale,
          this.anchor.y - pose[2][1] * this.scale,
          this.anchor.z
        );
      }
      addJoint(poseWorld[1], 1.1);
      addJoint(poseWorld[2], 1.1);

      // 3. Nose (Point 0): slightly forward like a real face
      poseWorld[0] = new THREE.Vector3(
        this.anchor.x + signX * pose[0][0] * this.scale,
        this.anchor.y - pose[0][1] * this.scale,
        this.anchor.z + 0.06
      );
      addJoint(poseWorld[0], 1.0);

      // 4. Biomechanical Forward Kinematic Depth for Left Arm (Elbow 3, Wrist 5)
      const L_upper = this.scale * 0.95;
      const L_lower = this.scale * 0.90;

      // Left Arm
      const isLActive = Math.abs(pose[5][0]) > 1e-4 || Math.abs(pose[5][1]) > 1e-4;
      if (isLActive) {
        const nx_e = pose[3][0];
        const ny_e = pose[3][1];
        const nx_w = pose[5][0];
        const ny_w = pose[5][1];

        const d2D_upper = Math.hypot(nx_e - pose[1][0], ny_e - pose[1][1]) * this.scale;
        const d2D_lower = Math.hypot(nx_w - nx_e, ny_w - ny_e) * this.scale;

        const isRaised = ny_w < 0.6 || Math.abs(nx_w) < 0.65;
        let elbowForwardZ = isRaised
          ? Math.max(0.06, Math.sqrt(Math.max(0, L_upper * L_upper - d2D_upper * d2D_upper)) * 0.65)
          : 0.02;
        let wristForwardZ = isRaised
          ? elbowForwardZ + Math.max(0.10, Math.sqrt(Math.max(0, L_lower * L_lower - d2D_lower * d2D_lower)) * 0.85)
          : 0.04;

        const worldElbowL = new THREE.Vector3(
          this.anchor.x + signX * nx_e * this.scale,
          this.anchor.y - ny_e * this.scale,
          this.anchor.z + elbowForwardZ
        );

        const worldWristL = new THREE.Vector3(
          this.anchor.x + signX * nx_w * this.scale,
          this.anchor.y - ny_w * this.scale,
          this.anchor.z + wristForwardZ
        );

        // Anti-Torso-Clipping hard collision floor in front of chest/sweater
        if (Math.abs(worldWristL.x) < 0.22 && worldWristL.y > 0.95) {
          worldElbowL.z = Math.max(worldElbowL.z, this.anchor.z + 0.14);
          worldWristL.z = Math.max(worldWristL.z, this.anchor.z + 0.22);
        }

        poseWorld[3] = worldElbowL;
        poseWorld[5] = worldWristL;
        addJoint(worldElbowL, 0.9);
        addJoint(worldWristL, 1.0);
      } else {
        poseWorld[3] = null;
        poseWorld[5] = null;
      }

      // Right Arm (Elbow 4, Wrist 6)
      const isRActive = Math.abs(pose[6][0]) > 1e-4 || Math.abs(pose[6][1]) > 1e-4;
      if (isRActive) {
        const nx_e = pose[4][0];
        const ny_e = pose[4][1];
        const nx_w = pose[6][0];
        const ny_w = pose[6][1];

        const d2D_upper = Math.hypot(nx_e - pose[2][0], ny_e - pose[2][1]) * this.scale;
        const d2D_lower = Math.hypot(nx_w - nx_e, ny_w - ny_e) * this.scale;

        const isRaised = ny_w < 0.6 || Math.abs(nx_w) < 0.65;
        let elbowForwardZ = isRaised
          ? Math.max(0.06, Math.sqrt(Math.max(0, L_upper * L_upper - d2D_upper * d2D_upper)) * 0.65)
          : 0.02;
        let wristForwardZ = isRaised
          ? elbowForwardZ + Math.max(0.10, Math.sqrt(Math.max(0, L_lower * L_lower - d2D_lower * d2D_lower)) * 0.85)
          : 0.04;

        const worldElbowR = new THREE.Vector3(
          this.anchor.x + signX * nx_e * this.scale,
          this.anchor.y - ny_e * this.scale,
          this.anchor.z + elbowForwardZ
        );

        const worldWristR = new THREE.Vector3(
          this.anchor.x + signX * nx_w * this.scale,
          this.anchor.y - ny_w * this.scale,
          this.anchor.z + wristForwardZ
        );

        // Anti-Torso-Clipping hard collision floor in front of chest/sweater
        if (Math.abs(worldWristR.x) < 0.22 && worldWristR.y > 0.95) {
          worldElbowR.z = Math.max(worldElbowR.z, this.anchor.z + 0.14);
          worldWristR.z = Math.max(worldWristR.z, this.anchor.z + 0.22);
        }

        poseWorld[4] = worldElbowR;
        poseWorld[6] = worldWristR;
        addJoint(worldElbowR, 0.9);
        addJoint(worldWristR, 1.0);
      } else {
        poseWorld[4] = null;
        poseWorld[6] = null;
      }

      // Draw Pose lines
      for (const [idxA, idxB] of this.poseEdges) {
        if (poseWorld[idxA] && poseWorld[idxB]) {
          addLine(poseWorld[idxA], poseWorld[idxB]);
        }
      }
    }

    // 5. Volumetric 3D Hand Depth (Fix 2: Never a Flat 2D Hand!)
    const renderHandVolumetric = (handPoints, wristAnchorWorld) => {
      if (!handPoints || handPoints.length < 21 || !wristAnchorWorld) return null;

      const handWristRaw = handPoints[0];
      const handWorld = [];

      // Palm transverse arch depth offsets relative to wrist
      const mcpArchOffsets = {
        1: 0.025, // Thumb CMC
        2: 0.040, // Thumb MCP
        5: 0.015, // Index MCP
        9: 0.022, // Middle MCP
        13: 0.015, // Ring MCP
        17: 0.005  // Pinky MCP
      };

      // Finger chain definitions [MCP, PIP, DIP, TIP]
      const fingerChains = [
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20]
      ];

      // Calculate finger curl ratios in 2D
      const fingerCurled = {};
      fingerChains.forEach((chain) => {
        const [mcp, pip, dip, tip] = chain;
        const direct = Math.hypot(handPoints[tip][0] - handPoints[mcp][0], handPoints[tip][1] - handPoints[mcp][1]);
        const curve = Math.max(
          Math.hypot(handPoints[pip][0] - handPoints[mcp][0], handPoints[pip][1] - handPoints[mcp][1]) +
          Math.hypot(handPoints[dip][0] - handPoints[pip][0], handPoints[dip][1] - handPoints[pip][1]) +
          Math.hypot(handPoints[tip][0] - handPoints[dip][0], handPoints[tip][1] - handPoints[dip][1]),
          0.001
        );
        const curl = THREE.MathUtils.clamp((0.85 - direct / curve) / (0.85 - 0.30), 0.0, 1.0);
        fingerCurled[mcp] = curl;
      });

      // Compute 3D world positions with volumetric palm arch and circular finger curl arcs
      for (let i = 0; i < 21; i++) {
        const pt = handPoints[i];
        const relX = pt[0] - handWristRaw[0];
        const relY = pt[1] - handWristRaw[1];

        // Base 3D Z offset relative to wrist
        let relZ = 0;

        // Arch offset
        if (mcpArchOffsets[i]) {
          relZ += mcpArchOffsets[i];
        }

        // Forward circular arc for curling fingers
        fingerChains.forEach((chain) => {
          const [mcp, pip, dip, tip] = chain;
          const curl = fingerCurled[mcp] || 0;
          const baseMcpZ = mcpArchOffsets[mcp] || 0.015;

          if (i === pip) {
            relZ = baseMcpZ + Math.sin(curl * 1.2) * 0.025;
          } else if (i === dip) {
            relZ = baseMcpZ + Math.sin(curl * 1.8) * 0.040;
          } else if (i === tip) {
            relZ = baseMcpZ + Math.sin(curl * 2.4) * 0.035;
          }
        });

        // Thumb forward curl arc (Points 3, 4)
        if (i === 3) {
          relZ = 0.035;
        } else if (i === 4) {
          relZ = 0.030;
        }

        const w = new THREE.Vector3(
          wristAnchorWorld.x + signX * relX * this.scale,
          wristAnchorWorld.y - relY * this.scale,
          wristAnchorWorld.z + relZ
        );

        handWorld[i] = w;
        addJoint(w, i === 0 ? 1.0 : 0.65);
      }

      // Draw Hand lines
      for (const [idxA, idxB] of this.handEdges) {
        if (handWorld[idxA] && handWorld[idxB]) {
          addLine(handWorld[idxA], handWorld[idxB]);
        }
      }

      return handWorld;
    };

    let leftHandWorld = null;
    if (leftHand && poseWorld[5]) {
      leftHandWorld = renderHandVolumetric(leftHand, poseWorld[5]);
    }

    let rightHandWorld = null;
    if (rightHand && poseWorld[6]) {
      rightHandWorld = renderHandVolumetric(rightHand, poseWorld[6]);
    }

    // Update GPU buffers
    this.lineGeometry.attributes.position.needsUpdate = true;
    this.lineGeometry.setDrawRange(0, vertexCount);
    this.jointMesh.count = jointCount;
    this.jointMesh.instanceMatrix.needsUpdate = true;

    return {
      poseWorld,
      leftHandWorld,
      rightHandWorld
    };
  }
}
