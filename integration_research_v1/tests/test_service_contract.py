from io import BytesIO

from PIL import Image

from pytorch_service.contract import build_response, validate_image_bytes


class StubTask:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def predict_image(self, image):
        self.calls += 1
        return self.values


def image_bytes():
    buf = BytesIO()
    Image.new('RGB', (32, 24), (20, 40, 60)).save(buf, format='PNG')
    return buf.getvalue()


def test_gate_below_threshold_skips_staging():
    binary = StubTask([0.5001, 0.4999])
    staging = StubTask([0.1, 0.2, 0.7])
    result = build_response(binary, staging, Image.open(BytesIO(image_bytes())), 1.0)
    assert result['final_stage'] == 0
    assert result['grading_performed'] is False
    assert result['conditional_stage_probabilities'] is None
    assert staging.calls == 0


def test_gate_at_threshold_runs_staging_and_uses_conditional_argmax():
    binary = StubTask([0.5, 0.5])
    staging = StubTask([0.51, 0.2, 0.29])
    result = build_response(binary, staging, Image.open(BytesIO(image_bytes())), 1.0)
    assert result['final_stage'] == 1
    assert result['grading_performed'] is True
    assert staging.calls == 1
    assert abs(sum(result['joint_stage_probabilities'].values()) - 1.0) < 1e-6


def test_gate_above_threshold_can_return_stage_three():
    binary = StubTask([0.1, 0.9])
    staging = StubTask([0.1, 0.2, 0.7])
    result = build_response(binary, staging, Image.open(BytesIO(image_bytes())), 1.0)
    assert result['final_stage'] == 3


def test_image_validation_rejects_signature_mismatch_and_large_dimensions():
    try:
        validate_image_bytes(b'not an image', 'image/png')
        raise AssertionError('expected invalid image rejection')
    except ValueError as exc:
        assert 'signature' in str(exc).lower() or 'decode' in str(exc).lower()

    buf = BytesIO()
    Image.new('RGB', (9000, 2)).save(buf, format='PNG')
    try:
        validate_image_bytes(buf.getvalue(), 'image/png')
        raise AssertionError('expected dimension rejection')
    except ValueError as exc:
        assert 'dimension' in str(exc).lower()
