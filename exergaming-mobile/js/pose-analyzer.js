// Ported from exergaming-system/src/pose/analyzer.py — PoseAnalyzer.
// Same 3-point angle (law of cosines via dot product) and Euclidean distance
// math, kept numerically identical so results match the desktop app exactly.

export function calculateAngle(p1, p2, p3) {
  const ax = p1[0] - p2[0], ay = p1[1] - p2[1];
  const cx = p3[0] - p2[0], cy = p3[1] - p2[1];

  const dot = ax * cx + ay * cy;
  const magA = Math.hypot(ax, ay);
  const magC = Math.hypot(cx, cy);

  if (magA === 0 || magC === 0) return 0;

  let cosine = dot / (magA * magC);
  cosine = Math.max(-1.0, Math.min(1.0, cosine)); // clamp, matches np.clip
  return (Math.acos(cosine) * 180) / Math.PI;
}

export function calculateDistance(p1, p2) {
  return Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
}
