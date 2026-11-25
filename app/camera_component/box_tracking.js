/* Keep a visual lock on a detector-confirmed box between model replies. */
const BOX_MAX_AGE = 2500;
const BOX_MATCH_GRACE = 200;
const UNTRACKED_MAX_AGE = 650;

function boxOverlap(a, b) {
  const area = box => (box[2] - box[0]) * (box[3] - box[1]);
  const intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) *
    Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return intersection / (area(a) + area(b) - intersection);
}

function boxTemplate(frame, box, margin = .06) {
  const {width, height, pixels} = frame;
  const sampleX = [], sampleY = [], values = [];
  for (let row = 0; row < 12; row++) {
    for (let col = 0; col < 12; col++) {
      const x = Math.round((box[0] + (box[2] - box[0]) * (margin + (1 - 2 * margin) * col / 11)) * (width - 1));
      const y = Math.round((box[1] + (box[3] - box[1]) * (margin + (1 - 2 * margin) * row / 11)) * (height - 1));
      const cx = Math.max(0, Math.min(width - 1, x)), cy = Math.max(0, Math.min(height - 1, y));
      sampleX.push(cx);
      sampleY.push(cy);
      values.push(pixels[cy * width + cx]);
    }
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const centered = Float32Array.from(values, value => value - mean);
  const energy = centered.reduce((sum, value) => sum + value * value, 0);
  if (energy < 64 * values.length || sampleX[11] - sampleX[0] < 5 ||
      sampleY.at(-1) - sampleY[0] < 5) return null;
  return {sampleX, sampleY, centered, energy, mean, width, height};
}

class CameraBoxTrack {
  constructor(detection, frame, observedAt) {
    this.detection = detection;
    this.box = detection.box;
    this.template = boxTemplate(frame, this.box) || boxTemplate(frame, this.box, -.12);
    this.observedAt = observedAt;
    this.lastMatch = observedAt;
    this.startedAt = observedAt;
    this.originX = Math.round((this.box[0] + this.box[2]) * (frame.width - 1) / 2);
    this.originY = Math.round((this.box[1] + this.box[3]) * (frame.height - 1) / 2);
    this.x = this.originX;
    this.y = this.originY;
    this.vx = 0;
    this.vy = 0;
    this.scale = 1;
  }

  isFresh(now) {
    if (!this.template) return now - this.observedAt < UNTRACKED_MAX_AGE;
    return now - this.observedAt < BOX_MAX_AGE && now - this.lastMatch <= BOX_MATCH_GRACE;
  }

  follow(frame, now) {
    if (!this.template) return false;
    const {sampleX, sampleY, centered, energy, mean, width, height} = this.template;
    if (frame.width !== width || frame.height !== height) return false;
    const radius = now - this.lastMatch > 60 ? 16 : 10;
    const predictedX = this.x + this.vx, predictedY = this.y + this.vy;
    let best = null;
    const search = (scale, x, y, range, step) => {
      const offsetsX = sampleX.map(value => Math.round((value - this.originX) * scale));
      const offsetsY = sampleY.map(value => Math.round((value - this.originY) * scale));
      for (let cy = y - range; cy <= y + range; cy += step) {
        if (cy + offsetsY[0] < 0 || cy + offsetsY.at(-1) >= height) continue;
        for (let cx = x - range; cx <= x + range; cx += step) {
          if (cx + offsetsX[0] < 0 || cx + offsetsX[11] >= width) continue;
          let sum = 0, squares = 0, product = 0, residual = 0;
          for (let i = 0; i < centered.length; i++) {
            const pixel = frame.pixels[(cy + offsetsY[i]) * width + cx + offsetsX[i]];
            sum += pixel;
            squares += pixel * pixel;
            product += centered[i] * pixel;
            residual += Math.abs(pixel - (centered[i] + mean));
          }
          const variance = squares - sum * sum / centered.length;
          if (variance < energy * .35 || residual / centered.length > 34) continue;
          const correlation = product / Math.sqrt(energy * variance);
          const score = correlation - .0015 * (Math.abs(cx - predictedX) + Math.abs(cy - predictedY)) -
            .12 * Math.abs(scale - this.scale);
          if (!best || score > best.score) best = {score, x: cx, y: cy, scale};
        }
      }
    };
    for (const factor of [1, .90, 1.10]) {
      const scale = Math.max(.65, Math.min(1.65, this.scale * factor));
      search(scale, predictedX, predictedY, 2, 1);
      search(scale, predictedX, predictedY, radius, 2);
    }
    if (best) search(best.scale, best.x, best.y, 1, 1);
    if (!best || best.score < .76) {
      this.vx = 0;
      this.vy = 0;
      return false;
    }
    const velocity = delta => Math.max(-7, Math.min(7, delta));
    this.vx = velocity(best.x - this.x);
    this.vy = velocity(best.y - this.y);
    this.x = best.x;
    this.y = best.y;
    this.scale = best.scale;
    this.lastMatch = now;
    this.box = this.detection.box.map((value, i) => {
      const [center, origin, size] = i % 2 === 0
        ? [best.x, this.originX, width - 1] : [best.y, this.originY, height - 1];
      return Math.max(0, Math.min(1, center / size + best.scale * (value - origin / size)));
    });
    return true;
  }
}
