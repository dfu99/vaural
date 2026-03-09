import copy

import pytest
import torch
import torch.nn as nn

from config import Config
from components import Emitter, ActionToSignal, Environment, Receiver, Pipeline
from train import pretrain_receiver, train_emitter


@pytest.fixture
def cfg():
    return Config(
        receiver_epochs=200,
        emitter_epochs=300,
        receiver_samples=2000,
        emitter_samples=2000,
    )


@pytest.fixture
def action_to_signal(cfg):
    return ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=100)


@pytest.fixture
def environment(cfg):
    return Environment(cfg.signal_dim, seed=200)


@pytest.fixture
def receiver(cfg):
    return Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)


@pytest.fixture
def emitter(cfg):
    return Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)


# ── Component shape tests ──


class TestShapes:
    def test_emitter_output_shape(self, cfg, emitter):
        x = torch.randn(8, cfg.sound_dim)
        out = emitter(x)
        assert out.shape == (8, cfg.action_dim)

    def test_action_to_signal_output_shape(self, cfg, action_to_signal):
        x = torch.randn(8, cfg.action_dim)
        out = action_to_signal(x)
        assert out.shape == (8, cfg.signal_dim)

    def test_environment_output_shape(self, cfg, environment):
        x = torch.randn(8, cfg.signal_dim)
        out = environment(x)
        assert out.shape == (8, cfg.signal_dim)

    def test_receiver_output_shape(self, cfg, receiver):
        x = torch.randn(8, cfg.signal_dim)
        out = receiver(x)
        assert out.shape == (8, cfg.sound_dim)

    def test_pipeline_output_shape(self, cfg, emitter, action_to_signal, environment, receiver):
        pipeline = Pipeline(emitter, action_to_signal, environment, receiver)
        x = torch.randn(8, cfg.sound_dim)
        out = pipeline(x)
        assert out.shape == (8, cfg.sound_dim)

    def test_single_sample_batch(self, cfg, emitter, action_to_signal, environment, receiver):
        pipeline = Pipeline(emitter, action_to_signal, environment, receiver)
        x = torch.randn(1, cfg.sound_dim)
        out = pipeline(x)
        assert out.shape == (1, cfg.sound_dim)


# ── Fixed component properties ──


class TestFixedComponents:
    def test_action_to_signal_deterministic(self, cfg):
        a1 = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=100)
        a2 = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=100)
        assert torch.equal(a1.weight, a2.weight)

    def test_environment_deterministic(self, cfg):
        e1 = Environment(cfg.signal_dim, seed=200)
        e2 = Environment(cfg.signal_dim, seed=200)
        assert torch.equal(e1.weight, e2.weight)

    def test_different_seeds_differ(self, cfg):
        a1 = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=100)
        a2 = ActionToSignal(cfg.action_dim, cfg.signal_dim, seed=999)
        assert not torch.equal(a1.weight, a2.weight)

    def test_action_to_signal_buffers_no_grad(self, action_to_signal):
        assert not action_to_signal.weight.requires_grad

    def test_environment_buffers_no_grad(self, environment):
        assert not environment.weight.requires_grad

    def test_fixed_components_have_no_parameters(self, action_to_signal, environment):
        assert list(action_to_signal.parameters()) == []
        assert list(environment.parameters()) == []


# ── Pipeline trace ──


class TestPipelineTrace:
    def test_trace_keys(self, emitter, action_to_signal, environment, receiver):
        pipeline = Pipeline(emitter, action_to_signal, environment, receiver)
        x = torch.randn(2, 16)
        trace = pipeline.forward_trace(x)
        assert set(trace.keys()) == {"input", "action", "signal", "received", "decoded"}

    def test_trace_matches_forward(self, emitter, action_to_signal, environment, receiver):
        pipeline = Pipeline(emitter, action_to_signal, environment, receiver)
        x = torch.randn(2, 16)
        out = pipeline(x)
        trace = pipeline.forward_trace(x)
        assert torch.allclose(out, trace["decoded"], atol=1e-6)


# ── Zero / NaN safety ──


class TestEdgeCases:
    def test_zero_input_no_nan(self, emitter, action_to_signal, environment, receiver):
        pipeline = Pipeline(emitter, action_to_signal, environment, receiver)
        x = torch.zeros(4, 16)
        out = pipeline(x)
        assert not torch.isnan(out).any()

    def test_large_input_no_nan(self, emitter, action_to_signal, environment, receiver):
        pipeline = Pipeline(emitter, action_to_signal, environment, receiver)
        x = torch.randn(4, 16) * 100
        out = pipeline(x)
        assert not torch.isnan(out).any()


# ── Receiver pre-training ──


class TestReceiverPretraining:
    def test_receiver_learns_to_invert(self, cfg, action_to_signal, environment):
        torch.manual_seed(cfg.seed)
        rec = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
        losses = pretrain_receiver(action_to_signal, environment, rec, cfg)
        assert losses[-1] < 0.15, f"Receiver final loss too high: {losses[-1]}"

    def test_loss_decreases(self, cfg, action_to_signal, environment):
        torch.manual_seed(cfg.seed)
        rec = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
        losses = pretrain_receiver(action_to_signal, environment, rec, cfg)
        assert losses[-1] < losses[0], "Loss did not decrease during receiver training"


# ── Emitter training ──


class TestEmitterTraining:
    @pytest.fixture
    def trained_pipeline(self, cfg, action_to_signal, environment):
        torch.manual_seed(cfg.seed)
        rec = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
        pretrain_receiver(action_to_signal, environment, rec, cfg)
        rec.requires_grad_(False)
        rec.eval()

        emit = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        pipeline = Pipeline(emit, action_to_signal, environment, rec)
        receiver_state_before = copy.deepcopy(rec.state_dict())
        losses = train_emitter(pipeline, cfg)
        return pipeline, losses, receiver_state_before

    def test_emitter_loss_decreases(self, trained_pipeline):
        _, losses, _ = trained_pipeline
        assert losses[-1] < losses[0], "Loss did not decrease during emitter training"

    def test_emitter_reaches_low_loss(self, trained_pipeline):
        _, losses, _ = trained_pipeline
        assert losses[-1] < 0.2, f"Emitter final loss too high: {losses[-1]}"

    def test_receiver_params_unchanged(self, trained_pipeline):
        pipeline, _, state_before = trained_pipeline
        state_after = pipeline.receiver.state_dict()
        for key in state_before:
            assert torch.equal(state_before[key], state_after[key]), \
                f"Receiver param {key} changed during emitter training"

    def test_emitter_params_changed(self, trained_pipeline, cfg):
        pipeline, _, _ = trained_pipeline
        # Freshly initialized emitter for comparison
        torch.manual_seed(cfg.seed)
        fresh = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        for p_trained, p_fresh in zip(pipeline.emitter.parameters(), fresh.parameters()):
            assert not torch.equal(p_trained, p_fresh), "Emitter params did not change"

    def test_receiver_requires_grad_false(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        for p in pipeline.receiver.parameters():
            assert not p.requires_grad

    def test_emitter_has_gradients(self, trained_pipeline):
        pipeline, _, _ = trained_pipeline
        assert any(p.grad is not None for p in pipeline.emitter.parameters())


# ── End-to-end ──


class TestEndToEnd:
    def test_round_trip_low_mse(self, cfg, action_to_signal, environment):
        """Full pipeline round-trips with low MSE after both training phases."""
        torch.manual_seed(cfg.seed)
        rec = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
        pretrain_receiver(action_to_signal, environment, rec, cfg)
        rec.requires_grad_(False)

        emit = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        pipeline = Pipeline(emit, action_to_signal, environment, rec)
        train_emitter(pipeline, cfg)

        pipeline.eval()
        with torch.no_grad():
            test = torch.randn(50, cfg.sound_dim)
            decoded = pipeline(test)
            mse = nn.functional.mse_loss(decoded, test).item()

        assert mse < 0.5, f"End-to-end MSE too high: {mse}"

    def test_generalizes_to_uniform(self, cfg, action_to_signal, environment):
        """Test on uniform distribution (trained on normal)."""
        torch.manual_seed(cfg.seed)
        rec = Receiver(cfg.signal_dim, cfg.sound_dim, cfg.hidden_dim)
        pretrain_receiver(action_to_signal, environment, rec, cfg)
        rec.requires_grad_(False)

        emit = Emitter(cfg.sound_dim, cfg.action_dim, cfg.hidden_dim)
        pipeline = Pipeline(emit, action_to_signal, environment, rec)
        train_emitter(pipeline, cfg)

        pipeline.eval()
        with torch.no_grad():
            test = torch.rand(50, cfg.sound_dim) * 2 - 1  # Uniform [-1, 1]
            decoded = pipeline(test)
            mse = nn.functional.mse_loss(decoded, test).item()

        assert mse < 0.1, f"Uniform-distribution MSE too high: {mse}"
