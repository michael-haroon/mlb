"""Records the state of the DL live-serving path, which does not serve the model
we train.

Three facts, established 2026-08-30 by running the real production shapes:

1. `inference_engine` constructs `LiveGameModel` (line ~248), not the
   `GameTransformer` that `train_unified.py` trains. `GameTransformer` is
   instantiated nowhere outside training and diagnostics, so it has NO serving
   path at all.

2. `LiveGameModel.forward` crashes on the engine's own shapes. The engine builds
   `mask = torch.ones_like(values)` at width `feature_dim`, `forward` concatenates
   three 16-dim embeddings onto values, and then hands the *concatenated* tensor to
   `_LSTMEncoder` together with a mask still at the original width. `values * mask`
   is a hard dimension error. This is the same drift already flagged as
   values(88)/mask(40) in mlb_dl/tests/test_inference_engine.py.

3. `LiveGameModel.forward` never reads `weather_temporal`, which the engine sets on
   every batch (the as-of [7,99] decision row, refreshed hourly). So the entire
   live weather path is computed and discarded.

Nothing in production is currently broken by this: `live_daemon.py` does not import
the engine. But the "reprice markets in real time" half of the project goal cannot
be delivered until the engine is ported to GameTransformer, which is a scope
decision (it touches the batch contract, all 20 feature-store tensors, the
standardizer, and market derivation) and is therefore raised rather than done.

The two forward-looking tests are `xfail(strict=True)` ON PURPOSE: when someone
ports the engine, they will start passing and pytest will fail loudly, forcing the
markers -- and this file's premise -- to be revisited.
"""

import pytest
import torch

from mlb_dl.models import LiveGameModel

FEATURE_DIM = 40   # engine default: config.get("feature_dim", 40)
EMBED_DIM = 8      # engine default is 16; any value reproduces the drift


def _model():
    m = LiveGameModel(feature_dim=FEATURE_DIM, hidden_dim=32, dropout=0.0,
                      batter_buckets=512, pitcher_buckets=512,
                      pitch_type_buckets=64, embed_dim=EMBED_DIM)
    m.eval()
    return m


def _live_batch(B=2, T=12):
    """Exactly what inference_engine._build_batch produces, including the
    `mask = torch.ones_like(values)` line that causes the drift."""
    torch.manual_seed(0)
    values = torch.randn(B, T, FEATURE_DIM)
    return {
        "values": values,
        "mask": torch.ones_like(values),
        "padding": torch.ones(B, T, dtype=torch.float32),
        "batter_hashes": torch.randint(1, 512, (B, T)),
        "pitcher_hashes": torch.randint(1, 512, (B, T)),
        "pitch_type_hashes": torch.randint(1, 64, (B, T)),
    }


def test_live_model_crashes_on_the_engines_own_shapes():
    """Pins fact 2 with its precise trigger, so a future reader does not have to
    rediscover why the engine tests are xfailed.

    forward() concatenates 3 x EMBED_DIM onto values, making x width
    FEATURE_DIM + 3*EMBED_DIM, then passes `ones_like(batch["mask"])` -- still
    FEATURE_DIM wide -- as the encoder's mask.
    """
    m = _model()
    with pytest.raises(RuntimeError, match=r"size of tensor"):
        with torch.no_grad():
            m(_live_batch())

    expected_x = FEATURE_DIM + 3 * EMBED_DIM
    assert expected_x != FEATURE_DIM, (
        "if these ever match, the drift is gone and this test is obsolete"
    )


@pytest.mark.xfail(strict=True, reason="engine serves LiveGameModel, which has no "
                                       "weather input; as-of weather is wired into "
                                       "GameTransformer. Remove when ported.")
def test_live_serving_output_responds_to_weather():
    """Fact 3: physically absurd weather must move the quote.

    Fails today for fact 2 (the crash) and would still fail for fact 3 after the
    crash is fixed, because forward() never reads the key.
    """
    m = _model()
    b1 = _live_batch()
    b2 = {k: v.clone() for k, v in b1.items()}
    b1["weather_temporal"] = torch.zeros(2, 7, 99)
    b2["weather_temporal"] = torch.full((2, 7, 99), 50.0)   # absurd
    with torch.no_grad():
        o1, o2 = m(b1), m(b2)
    assert any(not torch.allclose(o1[k], o2[k])
               for k in o1 if isinstance(o1[k], torch.Tensor)), \
        "weather_temporal had no effect on any output"


@pytest.mark.xfail(strict=True, reason="GameTransformer has no serving path; "
                                       "inference_engine constructs LiveGameModel. "
                                       "Remove when ported.")
def test_engine_serves_the_model_we_train():
    """Fact 1: the trained model and the served model must be the same class."""
    import inspect

    from mlb_dl import inference_engine
    src = inspect.getsource(inference_engine)
    assert "GameTransformer(" in src, (
        "inference_engine does not instantiate GameTransformer, so every head we "
        "train beyond home_win/yrfi (extra_innings, player HR/SB/hits/SO/HRBI, "
        "NegBin totals) and the entire as-of weather channel are unservable."
    )
