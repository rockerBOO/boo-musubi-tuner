"""Tests for Explorative Modeling (XM): best-of-K training.

Reference: https://explorative-modeling.github.io/
"""

import pytest
import torch
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.training import parser_common
from musubi_tuner.training.timesteps import get_sigmas
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer

from boo_musubi_tuner.explorative_modeling.explorative_modeling import (
    _CONTINUOUS_T_SAMPLING_MODES,
    ExplorativeModelingMixin,
    _gather_winner,
    _select_winner,
    explorative_modeling_setup_parser,
)


def test_select_winner_is_per_example_not_per_batch():
    # K=3 candidates, B=2 examples. Different winner per example.
    loss_stack = torch.tensor(
        [
            [4.0, 1.0],  # candidate 0
            [1.0, 4.0],  # candidate 1
            [9.0, 9.0],  # candidate 2
        ]
    )
    winner = _select_winner(loss_stack)
    assert winner.tolist() == [1, 0]


def test_select_winner_all_same_candidate():
    loss_stack = torch.tensor([[5.0, 5.0], [1.0, 1.0], [9.0, 9.0]])
    winner = _select_winner(loss_stack)
    assert winner.tolist() == [1, 1]


def test_gather_winner_selects_matching_candidate_1d():
    stack = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])  # (K=3, B=2)
    winner = torch.tensor([1, 0])
    gathered = _gather_winner(stack, winner)
    assert gathered.tolist() == [30.0, 20.0]


def test_gather_winner_multi_dim():
    stack = torch.arange(2 * 2 * 3).reshape(2, 2, 3).float()  # (K=2, B=2, C=3)
    winner = torch.tensor([1, 0])
    gathered = _gather_winner(stack, winner)
    assert gathered.shape == (2, 3)
    assert torch.equal(gathered[0], stack[1, 0])
    assert torch.equal(gathered[1], stack[0, 1])


def _xm_args(**overrides):
    parser = parser_common.setup_parser_common()
    parser = explorative_modeling_setup_parser(parser)
    args, _ = parser.parse_known_args([])
    args.timestep_sampling = "uniform"
    args.weighting_scheme = "none"
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class _FakeAccelerator:
    device = torch.device("cpu")


def _latents_and_noise(batch_size=2, shape=(4, 1, 2, 2)):
    latents = torch.zeros(batch_size, *shape)
    noise = torch.ones_like(latents)
    return latents, noise


class _ScriptedTrainer(ExplorativeModelingMixin, NetworkTrainer):
    """Test double: call_dit ignores its tensor inputs (beyond recording them)
    and returns a scripted per-example loss for each successive invocation,
    in call order. Lets us control exactly what each candidate's loss is
    without a real transformer.
    """

    def __init__(self, scripted_losses):
        super().__init__()
        self._scripted_losses = [torch.tensor(vals, dtype=torch.float32) for vals in scripted_losses]
        self.call_count = 0
        self.received_noise = []

    def call_dit(
        self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
    ):
        self.received_noise.append(noise.clone())
        vals = self._scripted_losses[self.call_count]
        self.call_count += 1
        shape = latents.shape
        pred = torch.sqrt(vals).view(-1, *([1] * (len(shape) - 1))).expand(shape).clone()
        target = torch.zeros_like(pred)
        return DiTOutput(pred=pred, target=target)


def test_process_batch_disabled_falls_through_to_base():
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(explorative_modeling=False)
    batch = {"timesteps": [0.3, 0.7]}
    trainer = _ScriptedTrainer([[4.0, 1.0]])

    _loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        None,
        None,
        batch,
        latents,
        noise,
        noise_scheduler,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert trainer.call_count == 1
    assert "xm/k" not in metrics


def test_process_batch_memory_efficient_scores_then_regenerates():
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(explorative_modeling=True, explorative_modeling_k=3, explorative_modeling_memory_efficient=True)
    batch = {"timesteps": [0.3, 0.7]}

    # K=3 scoring calls, per-example losses:
    #   ex0: [4, 1, 9] -> winner candidate 1 (loss 1)
    #   ex1: [1, 4, 9] -> winner candidate 0 (loss 1)
    scored = [[4.0, 1.0], [1.0, 4.0], [9.0, 9.0]]
    # 4th call: the memory-efficient regeneration forward on the gathered winners.
    regenerated = [1.0, 1.0]
    trainer = _ScriptedTrainer(scored + [regenerated])

    loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        None,
        None,
        batch,
        latents,
        noise,
        noise_scheduler,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert trainer.call_count == 4  # K scoring calls + 1 regeneration
    assert loss.item() == pytest.approx(1.0, rel=1e-4)
    assert metrics["xm/k"] == 3.0
    assert metrics["xm/candidate_loss_std"] > 0.0


def test_process_batch_reports_candidate_loss_distribution_metrics():
    # Same K=3 scoring setup as test_process_batch_memory_efficient_scores_then_regenerates:
    #   ex0 candidate losses: [4, 1, 9] -> per-example min=1, max=9, mean=14/3
    #   ex1 candidate losses: [1, 4, 9] -> per-example min=1, max=9, mean=14/3
    # Averaged over the batch: min=1.0, max=9.0, avg=14/3.
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(explorative_modeling=True, explorative_modeling_k=3, explorative_modeling_memory_efficient=True)
    batch = {"timesteps": [0.3, 0.7]}

    scored = [[4.0, 1.0], [1.0, 4.0], [9.0, 9.0]]
    regenerated = [1.0, 1.0]
    trainer = _ScriptedTrainer(scored + [regenerated])

    _, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        None,
        None,
        batch,
        latents,
        noise,
        noise_scheduler,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert metrics["xm/candidate_loss_min"] == pytest.approx(1.0, rel=1e-4)
    assert metrics["xm/candidate_loss_max"] == pytest.approx(9.0, rel=1e-4)
    assert metrics["xm/candidate_loss_avg"] == pytest.approx(14.0 / 3.0, rel=1e-4)


def test_process_batch_memory_efficient_regeneration_uses_gathered_winner_tensors():
    # Same K=3 scoring setup as test_process_batch_memory_efficient_scores_then_regenerates
    # (scored losses [[4,1],[1,4],[9,9]], winners [1, 0]), but this test proves the
    # regeneration forward actually receives the *gathered per-example winner* noise
    # tensors, not e.g. the original candidate-0 noise/noisy_1 reused by mistake.
    # _ScriptedTrainer.call_dit ignores its tensor args for computing loss (loss is
    # scripted by call order), so a bug that swaps in the wrong candidate tensors for
    # the regeneration forward would not change `loss` or `call_count` at all — only
    # inspecting the recorded `noise` tensors per call catches it.
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(explorative_modeling=True, explorative_modeling_k=3, explorative_modeling_memory_efficient=True)
    batch = {"timesteps": [0.3, 0.7]}

    scored = [[4.0, 1.0], [1.0, 4.0], [9.0, 9.0]]  # winners: ex0 -> candidate 1, ex1 -> candidate 0
    regenerated = [1.0, 1.0]
    trainer = _ScriptedTrainer(scored + [regenerated])

    torch.manual_seed(0)  # candidates 1 and 2 use torch.randn_like(latents); seed for reproducibility
    trainer.process_batch(
        args,
        _FakeAccelerator(),
        None,
        None,
        batch,
        latents,
        noise,
        noise_scheduler,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert trainer.call_count == 4
    candidate_0_noise, candidate_1_noise, candidate_2_noise, regeneration_noise = trainer.received_noise

    # Sanity: the three candidates' noise tensors must actually differ (otherwise this
    # test couldn't distinguish a correct gather from a wrong one).
    assert not torch.equal(candidate_0_noise, candidate_1_noise)
    assert not torch.equal(candidate_0_noise, candidate_2_noise)

    # Example 0's winner is candidate 1 -> regeneration's noise[0] must equal candidate 1's noise[0].
    assert torch.equal(regeneration_noise[0], candidate_1_noise[0])
    # Example 1's winner is candidate 0 -> regeneration's noise[1] must equal candidate 0's noise[1].
    assert torch.equal(regeneration_noise[1], candidate_0_noise[1])
    # And the regeneration forward must NOT just reuse candidate 0's (original) noise wholesale.
    assert not torch.equal(regeneration_noise[0], candidate_0_noise[0])


def test_process_batch_literal_mode_reuses_scored_forward_no_extra_call():
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(explorative_modeling=True, explorative_modeling_k=3, explorative_modeling_memory_efficient=False)
    batch = {"timesteps": [0.3, 0.7]}

    # Same per-example winners as above: ex0 -> candidate 1 (loss 1), ex1 -> candidate 0 (loss 1).
    scored = [[4.0, 1.0], [1.0, 4.0], [9.0, 9.0]]
    trainer = _ScriptedTrainer(scored)  # no extra scripted loss for a regeneration call

    loss, _metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        None,
        None,
        batch,
        latents,
        noise,
        noise_scheduler,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert trainer.call_count == 3  # no extra forward — literal mode gathers from the scored outputs
    # Winner losses are 1.0 and 1.0 for the two examples -> mean 1.0. A wrong gather
    # (e.g. picking candidate 0 for both) would give (4.0 + 1.0) / 2 = 2.5 instead.
    assert loss.item() == pytest.approx(1.0, rel=1e-4)


def test_process_batch_k1_matches_disabled_numerically():
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    batch = {"timesteps": [0.3, 0.7]}
    scripted = [[4.0, 1.0]]

    args_off = _xm_args(explorative_modeling=False)
    args_on = _xm_args(explorative_modeling=True, explorative_modeling_k=1, explorative_modeling_memory_efficient=False)

    trainer_off = _ScriptedTrainer(list(scripted))
    trainer_on = _ScriptedTrainer(list(scripted))

    loss_off, _ = trainer_off.process_batch(
        args_off,
        _FakeAccelerator(),
        None,
        None,
        batch,
        latents,
        noise,
        noise_scheduler,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    loss_on, _ = trainer_on.process_batch(
        args_on,
        _FakeAccelerator(),
        None,
        None,
        batch,
        latents,
        noise,
        noise_scheduler,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert loss_on.item() == pytest.approx(loss_off.item(), rel=1e-4)


class _DifferentiableTrainer(ExplorativeModelingMixin, NetworkTrainer):
    """Test double whose ``call_dit`` builds ``pred`` from a real ``nn.Parameter``
    via a differentiable op, so ``.backward()`` on the returned loss has somewhere
    real to flow. Used to prove neither scoring mode (memory-efficient / literal)
    accidentally leaves the final loss detached from the graph.
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def call_dit(
        self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
    ):
        pred = self.weight * noisy_model_input
        target = torch.zeros_like(pred)
        return DiTOutput(pred=pred, target=target)


@pytest.mark.parametrize("memory_efficient", [True, False])
def test_process_batch_backward_produces_gradients_both_modes(memory_efficient):
    latents, noise = _latents_and_noise()
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(explorative_modeling=True, explorative_modeling_k=3, explorative_modeling_memory_efficient=memory_efficient)
    batch = {"timesteps": [0.3, 0.7]}
    trainer = _DifferentiableTrainer()

    loss, _ = trainer.process_batch(
        args,
        _FakeAccelerator(),
        None,
        None,
        batch,
        latents,
        noise,
        noise_scheduler,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    loss.backward()

    assert trainer.weight.grad is not None
    assert trainer.weight.grad.abs().item() > 0.0


def test_memory_efficient_and_literal_modes_agree():
    # Design-doc requirement: memory-efficient and literal modes must select the SAME
    # per-example winner and produce a numerically equal loss/gradient for the same RNG-seeded
    # candidates -- the extra regeneration forward in memory-efficient mode should be exactly
    # equivalent to literal mode's gathered loss for the same winner, not just "close". Uses
    # `_DifferentiableTrainer` (real `nn.Parameter`, input-dependent `pred`) because
    # `_ScriptedTrainer`'s losses are keyed on call ORDER regardless of input tensors, so it
    # cannot prove the two modes agree on a real input-dependent computation.
    latents, noise = _latents_and_noise()
    batch = {"timesteps": [0.3, 0.7]}

    def run(memory_efficient):
        noise_scheduler = FlowMatchDiscreteScheduler()
        args = _xm_args(explorative_modeling=True, explorative_modeling_k=4, explorative_modeling_memory_efficient=memory_efficient)
        trainer = _DifferentiableTrainer()
        torch.manual_seed(1234)  # candidates 2..K use torch.randn_like -- seed identically per run
        loss, metrics = trainer.process_batch(
            args,
            _FakeAccelerator(),
            None,
            None,
            batch,
            latents,
            noise,
            noise_scheduler,
            torch.float32,
            torch.float32,
            None,
            0,
        )
        loss.backward()
        return loss, trainer.weight.grad.clone(), metrics

    loss_me, grad_me, metrics_me = run(memory_efficient=True)
    loss_lit, grad_lit, metrics_lit = run(memory_efficient=False)

    assert torch.allclose(loss_me, loss_lit, atol=1e-6)
    assert torch.allclose(grad_me, grad_lit, atol=1e-6)
    assert metrics_me["xm/candidate_loss_std"] == pytest.approx(metrics_lit["xm/candidate_loss_std"], abs=1e-6)


def test_extra_metadata_includes_xm_keys_when_enabled():
    trainer = _ScriptedTrainer([])
    args = _xm_args(explorative_modeling=True, explorative_modeling_k=5, explorative_modeling_memory_efficient=False)
    metadata = trainer.extra_metadata(args)
    assert metadata["ss_xm"] is True
    assert metadata["ss_xm_k"] == 5
    assert metadata["ss_xm_memory_efficient"] is False


def test_extra_metadata_empty_when_disabled():
    trainer = _ScriptedTrainer([])
    args = _xm_args(explorative_modeling=False)
    metadata = trainer.extra_metadata(args)
    assert metadata == {}


def test_on_train_start_logs_when_enabled(caplog):
    trainer = _ScriptedTrainer([])
    args = _xm_args(explorative_modeling=True, explorative_modeling_k=5, explorative_modeling_memory_efficient=False)
    with caplog.at_level("INFO", logger="boo_musubi_tuner.explorative_modeling.explorative_modeling"):
        trainer.on_train_start(args, _FakeAccelerator(), None, None, None)
    assert any("Explorative Modeling" in record.message for record in caplog.records)
    assert any("k=5" in record.message for record in caplog.records)
    assert any("memory_efficient=False" in record.message for record in caplog.records)


def test_on_train_start_silent_when_disabled(caplog):
    trainer = _ScriptedTrainer([])
    args = _xm_args(explorative_modeling=False)
    with caplog.at_level("INFO", logger="boo_musubi_tuner.explorative_modeling.explorative_modeling"):
        trainer.on_train_start(args, _FakeAccelerator(), None, None, None)
    assert caplog.records == []


def test_continuous_t_sampling_modes_matches_argparse_choices_minus_sigma():
    # Drift guard: `_CONTINUOUS_T_SAMPLING_MODES` is a hand-copied mirror of the dispatch
    # condition in `NetworkTrainer.get_noisy_model_input_and_timesteps` (which modes take the
    # continuous-t path vs. the on-schedule "sigma" path). If a new `--timestep_sampling` mode
    # is added to parser_common.py's `choices` without updating this frozenset, candidates 2..K
    # silently fall into the wrong noising branch -- no test failure today without this guard.
    # Pull `choices` from a real, live parser (not a second hardcoded copy) so this test fails
    # the moment the two definitions actually diverge.
    parser = parser_common.setup_parser_common()
    timestep_sampling_action = next(action for action in parser._actions if "--timestep_sampling" in action.option_strings)
    argparse_choices = set(timestep_sampling_action.choices)

    assert argparse_choices - {"sigma"} == _CONTINUOUS_T_SAMPLING_MODES


def test_parser_defaults():
    parser = explorative_modeling_setup_parser(parser_common.setup_parser_common())
    args, _ = parser.parse_known_args([])
    assert args.explorative_modeling is False
    assert args.explorative_modeling_k == 4
    assert args.explorative_modeling_memory_efficient is True


class _NoisyInputRecordingTrainer(ExplorativeModelingMixin, NetworkTrainer):
    """Test double: `call_dit` records the exact `noise` / `noisy_model_input` /
    `timesteps` tensors it receives per call (in call order) and returns a
    constant scripted loss, so `process_batch` can run to completion without a
    real transformer while we inspect exactly what candidate noising produced.
    """

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.received_noise = []
        self.received_noisy = []
        self.received_timesteps = []

    def call_dit(
        self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
    ):
        self.call_count += 1
        self.received_noise.append(noise.clone())
        self.received_noisy.append(noisy_model_input.clone())
        self.received_timesteps.append(timesteps.clone())
        pred = torch.zeros_like(latents)
        target = torch.zeros_like(latents)
        return DiTOutput(pred=pred, target=target)


def test_process_batch_candidate2_matches_base_trainer_formula_continuous_family():
    # "uniform" belongs to the continuous-t family: `get_noisy_model_input_and_timesteps`
    # computes `t` in [0, 1] directly and returns `timesteps = t * 1000 + 1` (off-schedule).
    # The mixin must invert that exact transform for candidates 2..K.
    latents = torch.randn(2, 4, 1, 2, 2)
    noise_1 = torch.randn(2, 4, 1, 2, 2)
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(
        timestep_sampling="uniform",
        explorative_modeling=True,
        explorative_modeling_k=2,
        explorative_modeling_memory_efficient=False,
    )
    batch = {"timesteps": [0.3, 0.7]}
    trainer = _NoisyInputRecordingTrainer()

    trainer.process_batch(
        args,
        _FakeAccelerator(),
        None,
        None,
        batch,
        latents,
        noise_1,
        noise_scheduler,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert trainer.call_count == 2  # k=2, literal mode: no extra regeneration forward
    noise_2_actual = trainer.received_noise[1]
    noisy_2_actual = trainer.received_noisy[1]

    # Reference: call the base trainer's OWN `get_noisy_model_input_and_timesteps` a
    # second time, with candidate 2's actual noise. For "uniform", `batch["timesteps"]`
    # (org_timesteps) is consumed directly as `t` with no randomness involved, so passing
    # the same `batch["timesteps"]` again deterministically reproduces the same `t` (and
    # therefore the same `timesteps`) as candidate 1 used — this is exactly the base
    # trainer's own formula, used as ground truth for what candidate 2 SHOULD look like.
    reference_trainer = NetworkTrainer()
    noisy_2_reference, timesteps_reference = reference_trainer.get_noisy_model_input_and_timesteps(
        args, noise_2_actual, latents, batch["timesteps"], noise_scheduler, torch.device("cpu"), torch.float32
    )

    assert torch.equal(timesteps_reference, trainer.received_timesteps[1])
    assert torch.allclose(noisy_2_actual, noisy_2_reference, atol=1e-6)


def test_process_batch_candidate2_matches_base_trainer_formula_sigma_family():
    # "sigma" is the argparse DEFAULT: `get_noisy_model_input_and_timesteps` samples
    # discrete on-schedule timesteps directly (no `+1` offset; `sigma = timesteps / 1000`
    # exactly via `get_sigmas`). This is the family the first fix round got wrong (it
    # applied the continuous family's `(timesteps - 1) / 1000` inversion unconditionally,
    # which is off by the `-1` for this branch).
    latents = torch.randn(2, 4, 1, 2, 2)
    noise_1 = torch.randn(2, 4, 1, 2, 2)
    noise_scheduler = FlowMatchDiscreteScheduler()
    args = _xm_args(
        timestep_sampling="sigma",
        explorative_modeling=True,
        explorative_modeling_k=2,
        explorative_modeling_memory_efficient=False,
    )
    batch = {"timesteps": [0.3, 0.7]}
    trainer = _NoisyInputRecordingTrainer()

    trainer.process_batch(
        args,
        _FakeAccelerator(),
        None,
        None,
        batch,
        latents,
        noise_1,
        noise_scheduler,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert trainer.call_count == 2  # k=2, literal mode: no extra regeneration forward
    timesteps_shared = trainer.received_timesteps[0]
    assert torch.equal(timesteps_shared, trainer.received_timesteps[1])  # same t for both candidates
    noise_2_actual = trainer.received_noise[1]
    noisy_2_actual = trainer.received_noisy[1]

    # Reference: `get_noisy_model_input_and_timesteps` has no mechanism to force a
    # specific `timesteps` for the "sigma" branch (it always resamples via
    # `compute_density_for_timestep_sampling`, ignoring the `timesteps` argument for this
    # branch) — so we instead call `get_sigmas` directly, which IS the base trainer's own
    # formula for this branch (imported verbatim into trainer_base.py and used exactly as
    # `sigmas = get_sigmas(...); noisy = sigmas * noise + (1 - sigmas) * latents`), applied
    # to the ACTUAL shared `timesteps` the mixin produced and candidate 2's actual noise.
    sigmas_reference = get_sigmas(noise_scheduler, timesteps_shared, torch.device("cpu"), n_dim=latents.ndim, dtype=torch.float32)
    noisy_2_reference = sigmas_reference * noise_2_actual + (1.0 - sigmas_reference) * latents

    assert torch.allclose(noisy_2_actual, noisy_2_reference, atol=1e-6)


def test_parser_accepts_overrides():
    parser = explorative_modeling_setup_parser(parser_common.setup_parser_common())
    args, _ = parser.parse_known_args(
        ["--explorative_modeling", "--explorative_modeling_k", "8", "--no-explorative_modeling_memory_efficient"]
    )
    assert args.explorative_modeling is True
    assert args.explorative_modeling_k == 8
    assert args.explorative_modeling_memory_efficient is False
