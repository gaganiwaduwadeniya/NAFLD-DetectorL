import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PYTORCH_RESEARCH_CONTRACT_ID,
  validatePytorchResearchResponse,
  validateUploadedImage,
} from '../../src/server/pytorchContract.js';

const positive = {
  success: true,
  schema_version: 4,
  contract_id: PYTORCH_RESEARCH_CONTRACT_ID,
  provider: 'pytorch_research_v1',
  model_version: { binary: 'nafld-research-v1', staging: 'nafld-research-v1' },
  dataset_version: '48b42abbee97b892591054019cdb4f84a65af9e0cd88d09fcbf8b3ae2a8b9856',
  research_use_only: true,
  clinically_validated: false,
  probability_calibration_verified: false,
  input_mode: 'single_frame',
  final_stage: 1,
  final_stage_label: 'Stage 1 — NAFLD, Mild steatosis',
  binary_probabilities: { 'Non-NAFLD': 0.1, NAFLD: 0.9 },
  binary_threshold: 0.5,
  grading_performed: true,
  conditional_stage_probabilities: { '1': 0.6, '2': 0.3, '3': 0.1 },
  joint_stage_probabilities: { '0': 0.1, '1': 0.54, '2': 0.27, '3': 0.09 },
  model_bundle: {
    binary: { architecture: 'densenet121', members: 5, manifest_sha256: 'daa75075d2de25823a2648dcb061b273e46459513657014cb239b5b8d2975cd9' },
    staging: { architecture: 'efficientnet_b0', members: 5, manifest_sha256: '12a0894534756dd380d08a4bfe9a6fe8086abde1420fd068140e709d3d83a52a' },
  },
  service_inference_ms: 1234.5,
  explanation_available: true,
};

test('accepts a complete gated positive response', () => {
  assert.equal(validatePytorchResearchResponse(positive), null);
});

test('accepts Stage 0 only when grading was not executed', () => {
  const negative = {
    ...positive,
    final_stage: 0,
    final_stage_label: 'Stage 0 — Non-NAFLD / Normal',
    binary_probabilities: { 'Non-NAFLD': 0.75, NAFLD: 0.25 },
    grading_performed: false,
    conditional_stage_probabilities: null,
    joint_stage_probabilities: null,
  };
  assert.equal(validatePytorchResearchResponse(negative), null);
});

test('rejects threshold changes, conditional misuse, and inconsistent stage', () => {
  assert.match(validatePytorchResearchResponse({ ...positive, binary_threshold: 0.49 }) ?? '', /threshold/i);
  assert.match(validatePytorchResearchResponse({ ...positive, grading_performed: false }) ?? '', /grading/i);
  assert.match(validatePytorchResearchResponse({ ...positive, final_stage: 3, final_stage_label: 'Stage 3 — NAFLD, Severe steatosis' }) ?? '', /argmax/i);
});

test('rejects invalid image signatures even when MIME claims PNG', () => {
  assert.match(validateUploadedImage(Buffer.from('not an image'), 'image/png') ?? '', /signature/i);
});

test('accepts a structurally valid PNG signature and dimensions', () => {
  const png = Buffer.alloc(24);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(png);
  png.writeUInt32BE(640, 16);
  png.writeUInt32BE(480, 20);
  assert.equal(validateUploadedImage(png, 'image/png'), null);
});
