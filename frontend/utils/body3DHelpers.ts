import * as THREE from 'three';

// Body part configuration for 3D positioning
export interface BodyPart3DConfig {
  name: string;
  position: THREE.Vector3;
  defaultZ: number;
  widthScale: number;
  heightScale: number;
}

// Configuration for mapping body parts to 3D coordinates
export const BODY_PART_3D_CONFIG: Record<string, BodyPart3DConfig> = {
  head: { name: 'Head', position: new THREE.Vector3(0, 1.7, 0), defaultZ: 0.1, widthScale: 0.15, heightScale: 0.1 },
  face: { name: 'Face', position: new THREE.Vector3(0, 1.6, 0.12), defaultZ: 0.15, widthScale: 0.12, heightScale: 0.08 },
  neck: { name: 'Neck', position: new THREE.Vector3(0, 1.45, 0.05), defaultZ: 0.08, widthScale: 0.08, heightScale: 0.05 },
  chest: { name: 'Chest', position: new THREE.Vector3(0, 1.2, 0.12), defaultZ: 0.15, widthScale: 0.25, heightScale: 0.15 },
  abdomen: { name: 'Abdomen', position: new THREE.Vector3(0, 0.95, 0.1), defaultZ: 0.12, widthScale: 0.22, heightScale: 0.12 },
  back_upper: { name: 'Upper Back', position: new THREE.Vector3(0, 1.2, -0.12), defaultZ: -0.15, widthScale: 0.25, heightScale: 0.15 },
  back_lower: { name: 'Lower Back', position: new THREE.Vector3(0, 0.95, -0.1), defaultZ: -0.12, widthScale: 0.22, heightScale: 0.12 },
  left_shoulder: { name: 'Left Shoulder', position: new THREE.Vector3(-0.25, 1.35, 0), defaultZ: 0, widthScale: 0.1, heightScale: 0.08 },
  right_shoulder: { name: 'Right Shoulder', position: new THREE.Vector3(0.25, 1.35, 0), defaultZ: 0, widthScale: 0.1, heightScale: 0.08 },
  left_arm: { name: 'Left Upper Arm', position: new THREE.Vector3(-0.35, 1.15, 0), defaultZ: 0, widthScale: 0.08, heightScale: 0.15 },
  right_arm: { name: 'Right Upper Arm', position: new THREE.Vector3(0.35, 1.15, 0), defaultZ: 0, widthScale: 0.08, heightScale: 0.15 },
  left_forearm: { name: 'Left Forearm', position: new THREE.Vector3(-0.4, 0.9, 0.05), defaultZ: 0.05, widthScale: 0.06, heightScale: 0.12 },
  right_forearm: { name: 'Right Forearm', position: new THREE.Vector3(0.4, 0.9, 0.05), defaultZ: 0.05, widthScale: 0.06, heightScale: 0.12 },
  left_hand: { name: 'Left Hand', position: new THREE.Vector3(-0.45, 0.7, 0.08), defaultZ: 0.08, widthScale: 0.05, heightScale: 0.06 },
  right_hand: { name: 'Right Hand', position: new THREE.Vector3(0.45, 0.7, 0.08), defaultZ: 0.08, widthScale: 0.05, heightScale: 0.06 },
  groin: { name: 'Groin', position: new THREE.Vector3(0, 0.75, 0.05), defaultZ: 0.05, widthScale: 0.15, heightScale: 0.08 },
  left_thigh: { name: 'Left Thigh', position: new THREE.Vector3(-0.12, 0.55, 0.05), defaultZ: 0.06, widthScale: 0.1, heightScale: 0.15 },
  right_thigh: { name: 'Right Thigh', position: new THREE.Vector3(0.12, 0.55, 0.05), defaultZ: 0.06, widthScale: 0.1, heightScale: 0.15 },
  left_knee: { name: 'Left Knee', position: new THREE.Vector3(-0.12, 0.4, 0.06), defaultZ: 0.07, widthScale: 0.08, heightScale: 0.06 },
  right_knee: { name: 'Right Knee', position: new THREE.Vector3(0.12, 0.4, 0.06), defaultZ: 0.07, widthScale: 0.08, heightScale: 0.06 },
  left_leg: { name: 'Left Lower Leg', position: new THREE.Vector3(-0.12, 0.22, 0.03), defaultZ: 0.04, widthScale: 0.07, heightScale: 0.12 },
  right_leg: { name: 'Right Lower Leg', position: new THREE.Vector3(0.12, 0.22, 0.03), defaultZ: 0.04, widthScale: 0.07, heightScale: 0.12 },
  left_foot: { name: 'Left Foot', position: new THREE.Vector3(-0.12, 0.05, 0.05), defaultZ: 0.06, widthScale: 0.06, heightScale: 0.04 },
  right_foot: { name: 'Right Foot', position: new THREE.Vector3(0.12, 0.05, 0.05), defaultZ: 0.06, widthScale: 0.06, heightScale: 0.04 },
  buttock: { name: 'Buttock', position: new THREE.Vector3(0, 0.72, -0.1), defaultZ: -0.12, widthScale: 0.2, heightScale: 0.1 },
};

// Risk level colors
export const RISK_COLORS = {
  high: 0xdc3545,    // Red
  moderate: 0xffc107, // Yellow
  medium: 0xffc107,   // Yellow (alias)
  low: 0x28a745,      // Green
};

// Get color for risk level
export const getRiskColor = (riskLevel: string): number => {
  const level = riskLevel?.toLowerCase() || 'low';
  return RISK_COLORS[level as keyof typeof RISK_COLORS] || RISK_COLORS.low;
};

// Map 2D body map coordinates to 3D position
export const map2Dto3D = (
  x: number,
  y: number,
  bodyPart: string
): THREE.Vector3 => {
  const config = BODY_PART_3D_CONFIG[bodyPart?.toLowerCase()] || BODY_PART_3D_CONFIG.chest;

  // Normalize x, y from percentage (0-100 or 0-1) to -1 to 1 range
  const normalizedX = x > 1 ? (x - 50) / 50 : (x - 0.5) * 2;
  const normalizedY = y > 1 ? (50 - y) / 50 : (0.5 - y) * 2;

  // Calculate position based on body part config with offset
  const position = config.position.clone();
  position.x += normalizedX * config.widthScale * 0.5;
  position.y += normalizedY * config.heightScale * 0.5;
  position.z = config.defaultZ;

  return position;
};

// Get body part name from position (for raycasting)
export const getBodyPartFromPosition = (position: THREE.Vector3): string | null => {
  let closestPart: string | null = null;
  let closestDistance = Infinity;

  for (const [partId, config] of Object.entries(BODY_PART_3D_CONFIG)) {
    const distance = position.distanceTo(config.position);
    if (distance < closestDistance) {
      closestDistance = distance;
      closestPart = partId;
    }
  }

  return closestPart;
};

// Camera presets for different view angles
export interface CameraPreset {
  azimuth: number;  // Horizontal angle (radians)
  polar: number;    // Vertical angle (radians)
  radius: number;   // Distance from center
  target: THREE.Vector3;
}

export const CAMERA_PRESETS: Record<string, CameraPreset> = {
  front: { azimuth: 0, polar: Math.PI / 2, radius: 3, target: new THREE.Vector3(0, 1, 0) },
  back: { azimuth: Math.PI, polar: Math.PI / 2, radius: 3, target: new THREE.Vector3(0, 1, 0) },
  left: { azimuth: -Math.PI / 2, polar: Math.PI / 2, radius: 3, target: new THREE.Vector3(0, 1, 0) },
  right: { azimuth: Math.PI / 2, polar: Math.PI / 2, radius: 3, target: new THREE.Vector3(0, 1, 0) },
};

// Calculate camera position from spherical coordinates
export const getCameraPositionFromSpherical = (
  azimuth: number,
  polar: number,
  radius: number,
  target: THREE.Vector3 = new THREE.Vector3(0, 1, 0)
): THREE.Vector3 => {
  const x = radius * Math.sin(polar) * Math.sin(azimuth);
  const y = radius * Math.cos(polar) + target.y;
  const z = radius * Math.sin(polar) * Math.cos(azimuth);

  return new THREE.Vector3(x, y, z);
};

// Lesion marker interface
export interface LesionMarker3D {
  id: number;
  position: THREE.Vector3;
  bodyPart: string;
  riskLevel: 'low' | 'moderate' | 'high';
  predictedClass?: string;
  date: string;
}

// Transform API lesion data to 3D markers
export const transformLesionsTo3D = (analyses: any[]): LesionMarker3D[] => {
  return analyses
    .filter((a: any) => a.body_location)
    .map((a: any) => {
      let position: THREE.Vector3;

      // Check if we have 3D coordinates
      if (a.body_map_coordinates?.z !== undefined) {
        position = new THREE.Vector3(
          a.body_map_coordinates.x,
          a.body_map_coordinates.y,
          a.body_map_coordinates.z
        );
      } else {
        // Map from 2D coordinates
        const x = a.body_map_coordinates?.x ?? 0.5;
        const y = a.body_map_coordinates?.y ?? 0.5;
        position = map2Dto3D(x, y, a.body_location);

        // Add small random offset to prevent overlapping
        position.x += (Math.random() - 0.5) * 0.02;
        position.y += (Math.random() - 0.5) * 0.02;
        position.z += (Math.random() - 0.5) * 0.01;
      }

      return {
        id: a.id,
        position,
        bodyPart: a.body_location,
        riskLevel: (a.risk_level?.toLowerCase() || 'low') as 'low' | 'moderate' | 'high',
        predictedClass: a.predicted_class,
        date: a.created_at,
      };
    });
};

// Create lesion marker mesh
export const createLesionMarkerMesh = (marker: LesionMarker3D): THREE.Mesh => {
  const geometry = new THREE.SphereGeometry(0.025, 16, 16);
  const material = new THREE.MeshBasicMaterial({
    color: getRiskColor(marker.riskLevel),
    transparent: true,
    opacity: 0.9,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(marker.position);
  mesh.userData = {
    lesionId: marker.id,
    isLesionMarker: true,
    riskLevel: marker.riskLevel,
  };
  mesh.name = `lesion_${marker.id}`;

  return mesh;
};

// Create pulsing ring for high-risk markers
export const createPulsingRing = (position: THREE.Vector3): THREE.Mesh => {
  const geometry = new THREE.RingGeometry(0.03, 0.04, 32);
  const material = new THREE.MeshBasicMaterial({
    color: 0xff0000,
    transparent: true,
    opacity: 0.6,
    side: THREE.DoubleSide,
  });

  const ring = new THREE.Mesh(geometry, material);
  ring.position.copy(position);
  ring.userData = { isPulsingRing: true };

  return ring;
};

// Perform raycasting to find intersections
export const performRaycast = (
  screenX: number,
  screenY: number,
  screenWidth: number,
  screenHeight: number,
  camera: THREE.Camera,
  objects: THREE.Object3D[]
): THREE.Intersection[] => {
  // Convert screen coordinates to normalized device coordinates
  const mouse = new THREE.Vector2(
    (screenX / screenWidth) * 2 - 1,
    -(screenY / screenHeight) * 2 + 1
  );

  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(mouse, camera);

  return raycaster.intersectObjects(objects, true);
};
