const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const script = fs.readFileSync(path.join(__dirname, '../app/camera_component/box_tracking.js'), 'utf8');
const CameraBoxTrack = vm.runInNewContext(`${script}\nCameraBoxTrack`);
const width = 160, height = 90;
const box = (left, top, size) => [left / (width - 1), top / (height - 1),
  (left + size - 1) / (width - 1), (top + size - 1) / (height - 1)];

function frame(left, top, size, flat = false) {
  const pixels = new Uint8Array(width * height).fill(40);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const texture = 145 + 46 * Math.sin(11 * x / size) + 36 * Math.cos(8 * y / size);
      pixels[(y + top) * width + x + left] = flat ? 150 : Math.round(texture);
    }
  }
  return {pixels, width, height};
}

test('the box follows motion without detector updates and grows with the object', () => {
  const track = new CameraBoxTrack({box: box(28, 26, 25)}, frame(28, 26, 25), 0);
  for (let i = 1; i <= 40; i++) {
    const left = 28 + 2 * i, size = 25 + Math.floor(i / 4);
    assert.equal(track.follow(frame(left, 26, size), 33 * i), true);
    assert.equal(track.isFresh(33 * i), true);
    assert.ok(Math.abs(track.box[0] * (width - 1) - left) < 6);
  }
  assert.ok((track.box[2] - track.box[0]) * (width - 1) > 28);
});

test('brief visual losses survive, a removed object expires, and flat objects can use their contour', () => {
  const track = new CameraBoxTrack({box: box(28, 26, 25)}, frame(28, 26, 25, true), 0);
  assert.ok(track.template);
  assert.equal(track.follow(frame(31, 26, 25, true), 33), true);
  assert.equal(track.follow(frame(0, 0, 0), 66), false);
  assert.equal(track.isFresh(99), true);
  assert.equal(track.follow(frame(40, 26, 25, true), 132), true);
  assert.equal(track.follow(frame(0, 0, 0), 165), false);
  assert.equal(track.isFresh(333), false);
});
