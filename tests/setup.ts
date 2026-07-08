/**
 * Test setup — runs before every test file
 * Provides browser-like environment (jsdom) and shared mocks.
 */

// Polyfill structuredClone for older Node versions (used by ReplayBuffer for Float32Array copy)
if (typeof structuredClone === 'undefined') {
  globalThis.structuredClone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
}

// Chart.js needs canvas, but we only test the agent/reward/env logic, not the charts.
// jsdom provides HTMLCanvasElement but no real 2d context — provide a no-op stub for tests
// that accidentally import a UI module. UI itself is excluded from coverage.

// Add TextEncoder/TextDecoder polyfills (jsdom usually has them, but be safe)
if (typeof TextEncoder === 'undefined') {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  globalThis.TextEncoder = require('util').TextEncoder;
}
if (typeof TextDecoder === 'undefined') {
  globalThis.TextDecoder = require('util').TextDecoder;
}
