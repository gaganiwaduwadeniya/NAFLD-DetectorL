export const PYTORCH_RESEARCH_CONTRACT_ID = 'nafld-pytorch-research-v1-single-frame-v1';
export const PYTORCH_DATASET_VERSION = '48b42abbee97b892591054019cdb4f84a65af9e0cd88d09fcbf8b3ae2a8b9856';
export const BINARY_MANIFEST_SHA256 = 'daa75075d2de25823a2648dcb061b273e46459513657014cb239b5b8d2975cd9';
export const STAGING_MANIFEST_SHA256 = '12a0894534756dd380d08a4bfe9a6fe8086abde1420fd068140e709d3d83a52a';

export const STAGE_LABELS = {
  0: 'Stage 0 — Non-NAFLD / Normal',
  1: 'Stage 1 — NAFLD, Mild steatosis',
  2: 'Stage 2 — NAFLD, Moderate steatosis',
  3: 'Stage 3 — NAFLD, Severe steatosis',
} as const;

export interface PytorchResearchResponse {
  success: true;
  schema_version: 4;
  contract_id: typeof PYTORCH_RESEARCH_CONTRACT_ID;
  provider: 'pytorch_research_v1';
  model_version: { binary: 'nafld-research-v1'; staging: 'nafld-research-v1' };
  dataset_version: typeof PYTORCH_DATASET_VERSION;
  research_use_only: true;
  clinically_validated: false;
  probability_calibration_verified: false;
  input_mode: 'single_frame';
  final_stage: 0 | 1 | 2 | 3;
  final_stage_label: string;
  binary_probabilities: { 'Non-NAFLD': number; NAFLD: number };
  binary_threshold: 0.5;
  grading_performed: boolean;
  conditional_stage_probabilities: Record<'1' | '2' | '3', number> | null;
  joint_stage_probabilities: Record<'0' | '1' | '2' | '3', number> | null;
  model_bundle: {
    binary: { architecture: 'densenet121'; members: 5; manifest_sha256: string };
    staging: { architecture: 'efficientnet_b0'; members: 5; manifest_sha256: string };
  };
  service_inference_ms: number;
  explanation_available: boolean;
  input_sha256?: string;
  input_geometry?: { width: number; height: number; mode: string };
}

function finiteProbability(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
}

function sumNearOne(values: number[]): boolean {
  return Math.abs(values.reduce((total, value) => total + value, 0) - 1) <= 1e-5;
}

export function validatePytorchResearchResponse(data: any): string | null {
  if (!data || typeof data !== 'object') return 'Response is not a JSON object.';
  if (data.success !== true || data.schema_version !== 4) return 'Expected a successful schema version 4 response.';
  if (data.contract_id !== PYTORCH_RESEARCH_CONTRACT_ID || data.provider !== 'pytorch_research_v1') return 'Contract or provider identity mismatch.';
  if (data.dataset_version !== PYTORCH_DATASET_VERSION) return 'Dataset version mismatch.';
  if (data.research_use_only !== true || data.clinically_validated !== false || data.probability_calibration_verified !== false) return 'Research and validation flags are invalid.';
  if (data.input_mode !== 'single_frame') return 'Input mode must be single_frame.';
  if (data.binary_threshold !== 0.5) return 'The locked binary threshold must equal 0.5.';
  if (![0, 1, 2, 3].includes(data.final_stage)) return 'Final stage must be an integer from 0 to 3.';
  if (data.final_stage_label !== STAGE_LABELS[data.final_stage as 0 | 1 | 2 | 3]) return 'Final stage label does not match the class contract.';
  const binary = data.binary_probabilities;
  if (!binary || !finiteProbability(binary['Non-NAFLD']) || !finiteProbability(binary.NAFLD) || !sumNearOne([binary['Non-NAFLD'], binary.NAFLD])) return 'Binary probabilities are invalid.';
  const expectedPositive = binary.NAFLD >= 0.5;
  if (data.grading_performed !== expectedPositive) return 'Grading execution does not match the locked binary gate.';
  if (data.final_stage === 0) {
    if (expectedPositive || data.conditional_stage_probabilities !== null || data.joint_stage_probabilities !== null) return 'Stage 0 must skip grading and omit conditional/joint scores.';
  } else {
    if (!expectedPositive || !data.conditional_stage_probabilities || !data.joint_stage_probabilities) return 'Positive stages require grading and both score vectors.';
    const conditional = data.conditional_stage_probabilities;
    const conditionalValues = [conditional['1'], conditional['2'], conditional['3']];
    if (!conditionalValues.every(finiteProbability) || !sumNearOne(conditionalValues)) return 'Conditional grading probabilities are invalid.';
    const expectedStage = conditionalValues.indexOf(Math.max(...conditionalValues)) + 1;
    if (data.final_stage !== expectedStage) return 'Final stage does not match conditional grading argmax.';
    const joint = data.joint_stage_probabilities;
    const jointValues = [joint['0'], joint['1'], joint['2'], joint['3']];
    if (!jointValues.every(finiteProbability) || !sumNearOne(jointValues)) return 'Derived joint stage scores are invalid.';
    if (Math.abs(joint['0'] - binary['Non-NAFLD']) > 1e-5) return 'Derived Stage 0 score is inconsistent.';
    for (const stage of ['1', '2', '3'] as const) {
      if (Math.abs(joint[stage] - binary.NAFLD * conditional[stage]) > 1e-5) return 'Derived joint stage score is inconsistent.';
    }
  }
  if (data.model_bundle?.binary?.architecture !== 'densenet121' || data.model_bundle.binary.members !== 5 || data.model_bundle.binary.manifest_sha256 !== BINARY_MANIFEST_SHA256) return 'Binary model bundle identity mismatch.';
  if (data.model_bundle?.staging?.architecture !== 'efficientnet_b0' || data.model_bundle.staging.members !== 5 || data.model_bundle.staging.manifest_sha256 !== STAGING_MANIFEST_SHA256) return 'Staging model bundle identity mismatch.';
  if (!Number.isFinite(data.service_inference_ms) || data.service_inference_ms < 0) return 'Service inference latency is invalid.';
  return null;
}

export function parseImageDimensions(input: Buffer): { width?: number; height?: number } {
  if (input.length >= 24 && input.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) {
    return { width: input.readUInt32BE(16), height: input.readUInt32BE(20) };
  }
  if (input.length >= 4 && input[0] === 0xff && input[1] === 0xd8) {
    let offset = 2;
    while (offset < input.length - 9) {
      if (input[offset] !== 0xff) { offset += 1; continue; }
      const marker = input[offset + 1];
      if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb].includes(marker)) {
        return { height: input.readUInt16BE(offset + 5), width: input.readUInt16BE(offset + 7) };
      }
      if (marker === 0xd8 || marker === 0xd9 || (marker >= 0xd0 && marker <= 0xd7)) { offset += 2; continue; }
      const length = input.readUInt16BE(offset + 2);
      if (length < 2) break;
      offset += 2 + length;
    }
  }
  return {};
}

export function validateUploadedImage(input: Buffer, mimetype: string): string | null {
  const isPng = input.length >= 8 && input.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  const isJpeg = input.length >= 2 && input[0] === 0xff && input[1] === 0xd8;
  if (!['image/png', 'image/jpeg'].includes(mimetype)) return 'Only JPEG and PNG images are accepted.';
  if ((mimetype === 'image/png' && !isPng) || (mimetype === 'image/jpeg' && !isJpeg)) return 'Image MIME type and byte signature do not agree.';
  const { width, height } = parseImageDimensions(input);
  if (!width || !height) return 'Image dimensions could not be read safely.';
  if (width > 8192 || height > 8192 || width * height > 40_000_000) return 'Image dimensions exceed the safety limit.';
  return null;
}
