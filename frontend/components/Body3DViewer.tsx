import React, { useRef, useEffect, useCallback, useState } from 'react';
import { View, StyleSheet, Dimensions, Platform } from 'react-native';
import { GLView, ExpoWebGLRenderingContext } from 'expo-gl';
import { Renderer } from 'expo-three';
import * as THREE from 'three';
import {
  Gesture,
  GestureDetector,
  GestureHandlerRootView,
} from 'react-native-gesture-handler';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';
import {
  LesionMarker3D,
  CAMERA_PRESETS,
  getCameraPositionFromSpherical,
  getRiskColor,
  performRaycast,
  BODY_PART_3D_CONFIG,
} from '../utils/body3DHelpers';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');
const VIEWER_HEIGHT = SCREEN_HEIGHT * 0.5;

interface Body3DViewerProps {
  lesions: LesionMarker3D[];
  onLesionSelect?: (lesion: LesionMarker3D) => void;
  onBodyPartTap?: (bodyPart: string, position: THREE.Vector3) => void;
  selectedLesionId?: number | null;
  viewPreset?: 'front' | 'back' | 'left' | 'right' | 'free';
  showBodyPartLabels?: boolean;
}

// Create a stylized procedural human body using basic shapes
const createProceduralBody = (): THREE.Group => {
  const body = new THREE.Group();
  const bodyMaterial = new THREE.MeshPhongMaterial({
    color: 0xf5d0c5,
    shininess: 30,
    flatShading: false,
  });

  // Head
  const headGeometry = new THREE.SphereGeometry(0.12, 32, 32);
  const head = new THREE.Mesh(headGeometry, bodyMaterial.clone());
  head.position.set(0, 1.7, 0);
  head.name = 'head';
  body.add(head);

  // Neck
  const neckGeometry = new THREE.CylinderGeometry(0.05, 0.06, 0.1, 16);
  const neck = new THREE.Mesh(neckGeometry, bodyMaterial.clone());
  neck.position.set(0, 1.53, 0);
  neck.name = 'neck';
  body.add(neck);

  // Torso (chest + abdomen)
  const torsoGeometry = new THREE.CylinderGeometry(0.18, 0.15, 0.5, 16);
  const torso = new THREE.Mesh(torsoGeometry, bodyMaterial.clone());
  torso.position.set(0, 1.2, 0);
  torso.name = 'chest';
  body.add(torso);

  // Lower torso (abdomen)
  const abdomenGeometry = new THREE.CylinderGeometry(0.15, 0.12, 0.25, 16);
  const abdomen = new THREE.Mesh(abdomenGeometry, bodyMaterial.clone());
  abdomen.position.set(0, 0.85, 0);
  abdomen.name = 'abdomen';
  body.add(abdomen);

  // Hips
  const hipsGeometry = new THREE.SphereGeometry(0.15, 16, 16);
  hipsGeometry.scale(1, 0.5, 0.8);
  const hips = new THREE.Mesh(hipsGeometry, bodyMaterial.clone());
  hips.position.set(0, 0.72, 0);
  hips.name = 'groin';
  body.add(hips);

  // Shoulders
  const shoulderGeometry = new THREE.SphereGeometry(0.07, 16, 16);

  const leftShoulder = new THREE.Mesh(shoulderGeometry, bodyMaterial.clone());
  leftShoulder.position.set(-0.25, 1.4, 0);
  leftShoulder.name = 'left_shoulder';
  body.add(leftShoulder);

  const rightShoulder = new THREE.Mesh(shoulderGeometry, bodyMaterial.clone());
  rightShoulder.position.set(0.25, 1.4, 0);
  rightShoulder.name = 'right_shoulder';
  body.add(rightShoulder);

  // Upper Arms
  const upperArmGeometry = new THREE.CylinderGeometry(0.045, 0.04, 0.3, 12);

  const leftUpperArm = new THREE.Mesh(upperArmGeometry, bodyMaterial.clone());
  leftUpperArm.position.set(-0.32, 1.2, 0);
  leftUpperArm.rotation.z = 0.15;
  leftUpperArm.name = 'left_arm';
  body.add(leftUpperArm);

  const rightUpperArm = new THREE.Mesh(upperArmGeometry, bodyMaterial.clone());
  rightUpperArm.position.set(0.32, 1.2, 0);
  rightUpperArm.rotation.z = -0.15;
  rightUpperArm.name = 'right_arm';
  body.add(rightUpperArm);

  // Forearms
  const forearmGeometry = new THREE.CylinderGeometry(0.035, 0.03, 0.28, 12);

  const leftForearm = new THREE.Mesh(forearmGeometry, bodyMaterial.clone());
  leftForearm.position.set(-0.38, 0.92, 0.05);
  leftForearm.rotation.z = 0.1;
  leftForearm.rotation.x = -0.2;
  leftForearm.name = 'left_forearm';
  body.add(leftForearm);

  const rightForearm = new THREE.Mesh(forearmGeometry, bodyMaterial.clone());
  rightForearm.position.set(0.38, 0.92, 0.05);
  rightForearm.rotation.z = -0.1;
  rightForearm.rotation.x = -0.2;
  rightForearm.name = 'right_forearm';
  body.add(rightForearm);

  // Hands
  const handGeometry = new THREE.SphereGeometry(0.04, 12, 12);
  handGeometry.scale(0.8, 1.2, 0.5);

  const leftHand = new THREE.Mesh(handGeometry, bodyMaterial.clone());
  leftHand.position.set(-0.42, 0.72, 0.08);
  leftHand.name = 'left_hand';
  body.add(leftHand);

  const rightHand = new THREE.Mesh(handGeometry, bodyMaterial.clone());
  rightHand.position.set(0.42, 0.72, 0.08);
  rightHand.name = 'right_hand';
  body.add(rightHand);

  // Upper Legs (thighs)
  const thighGeometry = new THREE.CylinderGeometry(0.08, 0.06, 0.35, 16);

  const leftThigh = new THREE.Mesh(thighGeometry, bodyMaterial.clone());
  leftThigh.position.set(-0.1, 0.52, 0);
  leftThigh.name = 'left_thigh';
  body.add(leftThigh);

  const rightThigh = new THREE.Mesh(thighGeometry, bodyMaterial.clone());
  rightThigh.position.set(0.1, 0.52, 0);
  rightThigh.name = 'right_thigh';
  body.add(rightThigh);

  // Knees
  const kneeGeometry = new THREE.SphereGeometry(0.055, 12, 12);

  const leftKnee = new THREE.Mesh(kneeGeometry, bodyMaterial.clone());
  leftKnee.position.set(-0.1, 0.35, 0.02);
  leftKnee.name = 'left_knee';
  body.add(leftKnee);

  const rightKnee = new THREE.Mesh(kneeGeometry, bodyMaterial.clone());
  rightKnee.position.set(0.1, 0.35, 0.02);
  rightKnee.name = 'right_knee';
  body.add(rightKnee);

  // Lower Legs
  const lowerLegGeometry = new THREE.CylinderGeometry(0.05, 0.04, 0.35, 12);

  const leftLowerLeg = new THREE.Mesh(lowerLegGeometry, bodyMaterial.clone());
  leftLowerLeg.position.set(-0.1, 0.15, 0);
  leftLowerLeg.name = 'left_leg';
  body.add(leftLowerLeg);

  const rightLowerLeg = new THREE.Mesh(lowerLegGeometry, bodyMaterial.clone());
  rightLowerLeg.position.set(0.1, 0.15, 0);
  rightLowerLeg.name = 'right_leg';
  body.add(rightLowerLeg);

  // Feet
  const footGeometry = new THREE.BoxGeometry(0.06, 0.04, 0.12);

  const leftFoot = new THREE.Mesh(footGeometry, bodyMaterial.clone());
  leftFoot.position.set(-0.1, 0, 0.03);
  leftFoot.name = 'left_foot';
  body.add(leftFoot);

  const rightFoot = new THREE.Mesh(footGeometry, bodyMaterial.clone());
  rightFoot.position.set(0.1, 0, 0.03);
  rightFoot.name = 'right_foot';
  body.add(rightFoot);

  // Back regions (invisible, for raycasting)
  const backUpperGeometry = new THREE.BoxGeometry(0.35, 0.3, 0.05);
  const backMaterial = new THREE.MeshBasicMaterial({ visible: false });

  const backUpper = new THREE.Mesh(backUpperGeometry, backMaterial);
  backUpper.position.set(0, 1.2, -0.15);
  backUpper.name = 'back_upper';
  body.add(backUpper);

  const backLower = new THREE.Mesh(backUpperGeometry.clone(), backMaterial.clone());
  backLower.position.set(0, 0.9, -0.12);
  backLower.name = 'back_lower';
  body.add(backLower);

  return body;
};

// Create lesion marker mesh
const createMarkerMesh = (marker: LesionMarker3D, isSelected: boolean): THREE.Group => {
  const group = new THREE.Group();

  // Main sphere
  const geometry = new THREE.SphereGeometry(isSelected ? 0.035 : 0.025, 16, 16);
  const material = new THREE.MeshBasicMaterial({
    color: getRiskColor(marker.riskLevel),
    transparent: true,
    opacity: 0.9,
  });

  const sphere = new THREE.Mesh(geometry, material);
  sphere.userData = {
    lesionId: marker.id,
    isLesionMarker: true,
    riskLevel: marker.riskLevel,
  };

  group.add(sphere);

  // Add ring for high-risk markers
  if (marker.riskLevel === 'high') {
    const ringGeometry = new THREE.RingGeometry(0.035, 0.045, 32);
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: 0xff0000,
      transparent: true,
      opacity: 0.6,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeometry, ringMaterial);
    ring.userData = { isPulsingRing: true };
    group.add(ring);
  }

  group.position.copy(marker.position);
  group.name = `lesion_${marker.id}`;
  group.userData = { lesionId: marker.id, isLesionMarker: true };

  return group;
};

export const Body3DViewer: React.FC<Body3DViewerProps> = ({
  lesions,
  onLesionSelect,
  onBodyPartTap,
  selectedLesionId,
  viewPreset = 'front',
  showBodyPartLabels = false,
}) => {
  // Refs for Three.js objects
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<Renderer | null>(null);
  const bodyRef = useRef<THREE.Group | null>(null);
  const markersRef = useRef<THREE.Group | null>(null);
  const glRef = useRef<ExpoWebGLRenderingContext | null>(null);
  const frameIdRef = useRef<number | null>(null);

  // Camera state using shared values for smooth animations
  const azimuth = useSharedValue(CAMERA_PRESETS[viewPreset]?.azimuth ?? 0);
  const polar = useSharedValue(CAMERA_PRESETS[viewPreset]?.polar ?? Math.PI / 2);
  const radius = useSharedValue(CAMERA_PRESETS[viewPreset]?.radius ?? 3);

  // Animation time for pulsing effects
  const animationTime = useRef(0);

  // State for highlighting
  const [highlightedBodyPart, setHighlightedBodyPart] = useState<string | null>(null);

  // Initialize Three.js scene
  const onContextCreate = useCallback(async (gl: ExpoWebGLRenderingContext) => {
    glRef.current = gl;

    // Create renderer
    const renderer = new Renderer({ gl });
    renderer.setSize(gl.drawingBufferWidth, gl.drawingBufferHeight);
    renderer.setClearColor(0x1a1a2e, 1);
    rendererRef.current = renderer;

    // Create scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      45,
      gl.drawingBufferWidth / gl.drawingBufferHeight,
      0.1,
      100
    );
    cameraRef.current = camera;

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(-5, 3, -5);
    scene.add(backLight);

    // Create body model
    const body = createProceduralBody();
    bodyRef.current = body;
    scene.add(body);

    // Create markers group
    const markersGroup = new THREE.Group();
    markersRef.current = markersGroup;
    scene.add(markersGroup);

    // Add floor grid for reference
    const gridHelper = new THREE.GridHelper(2, 10, 0x444444, 0x333333);
    gridHelper.position.y = -0.02;
    scene.add(gridHelper);

    // Start render loop
    const render = () => {
      frameIdRef.current = requestAnimationFrame(render);

      // Update camera position from spherical coordinates
      const cameraPos = getCameraPositionFromSpherical(
        azimuth.value,
        polar.value,
        radius.value
      );
      camera.position.copy(cameraPos);
      camera.lookAt(0, 1, 0);

      // Update pulsing animation
      animationTime.current += 0.05;
      markersGroup.children.forEach((child) => {
        if (child instanceof THREE.Group) {
          child.children.forEach((mesh) => {
            if (mesh.userData?.isPulsingRing) {
              const scale = 1 + Math.sin(animationTime.current) * 0.2;
              mesh.scale.set(scale, scale, 1);
              (mesh as THREE.Mesh).material.opacity = 0.3 + Math.sin(animationTime.current) * 0.3;
            }
          });
        }
      });

      renderer.render(scene, camera);
      gl.endFrameEXP();
    };

    render();
  }, []);

  // Update markers when lesions change
  useEffect(() => {
    if (!markersRef.current) return;

    // Clear existing markers
    while (markersRef.current.children.length > 0) {
      const child = markersRef.current.children[0];
      markersRef.current.remove(child);
    }

    // Add new markers
    lesions.forEach((lesion) => {
      const isSelected = lesion.id === selectedLesionId;
      const markerGroup = createMarkerMesh(lesion, isSelected);
      markersRef.current?.add(markerGroup);
    });
  }, [lesions, selectedLesionId]);

  // Update camera when viewPreset changes
  useEffect(() => {
    const preset = CAMERA_PRESETS[viewPreset];
    if (preset) {
      azimuth.value = withSpring(preset.azimuth, { damping: 15 });
      polar.value = withSpring(preset.polar, { damping: 15 });
      radius.value = withSpring(preset.radius, { damping: 15 });
    }
  }, [viewPreset]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (frameIdRef.current) {
        cancelAnimationFrame(frameIdRef.current);
      }
      // Dispose of Three.js resources
      if (sceneRef.current) {
        sceneRef.current.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            object.geometry.dispose();
            if (Array.isArray(object.material)) {
              object.material.forEach((m) => m.dispose());
            } else {
              object.material.dispose();
            }
          }
        });
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  // Handle tap for lesion/body part selection
  const handleTap = useCallback(
    (x: number, y: number) => {
      if (!cameraRef.current || !sceneRef.current || !glRef.current) return;

      const gl = glRef.current;
      const viewerWidth = gl.drawingBufferWidth;
      const viewerHeight = gl.drawingBufferHeight;

      // Collect all raycastable objects
      const objects: THREE.Object3D[] = [];
      if (markersRef.current) {
        markersRef.current.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            objects.push(child);
          }
        });
      }
      if (bodyRef.current) {
        bodyRef.current.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            objects.push(child);
          }
        });
      }

      const intersects = performRaycast(
        x,
        y,
        viewerWidth,
        viewerHeight,
        cameraRef.current,
        objects
      );

      if (intersects.length > 0) {
        const hit = intersects[0];

        // Check if we hit a lesion marker
        let current: THREE.Object3D | null = hit.object;
        while (current) {
          if (current.userData?.isLesionMarker) {
            const lesionId = current.userData.lesionId;
            const lesion = lesions.find((l) => l.id === lesionId);
            if (lesion && onLesionSelect) {
              onLesionSelect(lesion);
            }
            return;
          }
          current = current.parent;
        }

        // If not a marker, check body part
        const bodyPartName = hit.object.name;
        if (bodyPartName && onBodyPartTap) {
          onBodyPartTap(bodyPartName, hit.point);
        }
      }
    },
    [lesions, onLesionSelect, onBodyPartTap]
  );

  // Gesture handlers
  const panGesture = Gesture.Pan()
    .onUpdate((e) => {
      // Rotate camera
      azimuth.value += e.velocityX * 0.0001;
      const newPolar = polar.value - e.velocityY * 0.0001;
      polar.value = Math.max(0.3, Math.min(Math.PI - 0.3, newPolar));
    })
    .minDistance(5);

  const pinchGesture = Gesture.Pinch()
    .onUpdate((e) => {
      const newRadius = radius.value / e.scale;
      radius.value = Math.max(1.5, Math.min(6, newRadius));
    });

  const tapGesture = Gesture.Tap()
    .onEnd((e) => {
      runOnJS(handleTap)(e.x, e.y);
    });

  const composedGestures = Gesture.Simultaneous(
    Gesture.Race(tapGesture, panGesture),
    pinchGesture
  );

  return (
    <View style={styles.container}>
      <GestureDetector gesture={composedGestures}>
        <View style={styles.glContainer}>
          <GLView
            style={styles.glView}
            onContextCreate={onContextCreate}
          />
        </View>
      </GestureDetector>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    width: SCREEN_WIDTH - 32,
    height: VIEWER_HEIGHT,
    borderRadius: 16,
    overflow: 'hidden',
    backgroundColor: '#1a1a2e',
  },
  glContainer: {
    flex: 1,
  },
  glView: {
    flex: 1,
  },
});

export default Body3DViewer;
