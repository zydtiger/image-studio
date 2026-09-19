"""Subprocess-only check against real inference packages, using tiny CPU tensors.

No pretrained weights, CUDA initialization, Hub calls or model downloads.
Keeping heavy imports here preserves the API-process import-purity test.
"""

import json
import sys
from pathlib import Path


def main(root: Path):
    import torch
    from diffusers import AnimaTextConditioner, ClassifierFreeGuidance, CosmosTransformer3DModel
    from diffusers.modular_pipelines.modular_pipeline import BlockState

    from image_studio.inference.anima import load_checkpoint_models, make_pipeline
    from image_studio.inference.z_image import GenerationCancelled

    # Independent inverse mapping produces original-format synthetic weights.
    inverse = (
        ("time_embed.t_embedder", "t_embedder.1"),
        ("time_embed.norm", "t_embedding_norm"),
        ("transformer_blocks", "blocks"),
        ("norm1.linear_1", "adaln_modulation_self_attn.1"),
        ("norm1.linear_2", "adaln_modulation_self_attn.2"),
        ("norm2.linear_1", "adaln_modulation_cross_attn.1"),
        ("norm2.linear_2", "adaln_modulation_cross_attn.2"),
        ("norm3.linear_1", "adaln_modulation_mlp.1"),
        ("norm3.linear_2", "adaln_modulation_mlp.2"),
        ("attn1", "self_attn"),
        ("attn2", "cross_attn"),
        ("to_q", "q_proj"),
        ("to_k", "k_proj"),
        ("to_v", "v_proj"),
        ("to_out.0", "output_proj"),
        ("norm_q", "q_norm"),
        ("norm_k", "k_norm"),
        ("ff.net.0.proj", "mlp.layer1"),
        ("ff.net.2", "mlp.layer2"),
        ("patch_embed.proj", "x_embedder.proj.1"),
        ("norm_out.linear_1", "final_layer.adaln_modulation.1"),
        ("norm_out.linear_2", "final_layer.adaln_modulation.2"),
        ("proj_out", "final_layer.linear"),
    )
    torch.set_num_threads(1)
    for layers in (28, 40):
        transformer = CosmosTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=1,
            attention_head_dim=16,
            num_layers=layers,
            mlp_ratio=1,
            text_embed_dim=8,
            adaln_lora_dim=4,
            max_size=(2, 4, 4),
            extra_pos_embed_type=None,
        )
        conditioner = AnimaTextConditioner(
            source_dim=8,
            target_dim=8,
            model_dim=8,
            num_layers=1,
            num_attention_heads=1,
            mlp_ratio=1,
            target_vocab_size=16,
            min_sequence_length=4,
        )
        for name, model in (("transformer", transformer), ("text_conditioner", conditioner)):
            (root / name).mkdir(exist_ok=True)
            (root / name / "config.json").write_text(json.dumps(dict(model.config)))
        original = {}
        for key, value in transformer.state_dict().items():
            for old, new in inverse:
                key = key.replace(old, new)
            original["net." + key] = value.clone()
        original.update(
            {"net.llm_adapter." + k: v.clone() for k, v in conditioner.state_dict().items()}
        )
        actual, actual_conditioner = load_checkpoint_models(original, root, layers, torch.float32)
        assert actual.config.num_layers == layers
        for expected_model, actual_model in (
            (transformer, actual),
            (conditioner, actual_conditioner),
        ):
            for key, value in expected_model.state_dict().items():
                assert torch.equal(value, actual_model.state_dict()[key]), key
                assert actual_model.state_dict()[key].device.type == "cpu"
        # Strict key and shape failures: no silent partial/random initialization.
        for bad in (
            {k: v for k, v in original.items() if k != "net.x_embedder.proj.1.weight"},
            {**original, "net.x_embedder.proj.1.weight": torch.zeros(1)},
            {**original, "net.unexpected.weight": torch.zeros(1)},
        ):
            try:
                load_checkpoint_models(bad, root, layers, torch.float32)
            except RuntimeError:
                pass
            else:
                raise AssertionError("malformed checkpoint accepted")

    loaded = make_pipeline()
    loaded.pipeline.update_components(guider=ClassifierFreeGuidance(guidance_scale=1))
    assert loaded.pipeline.guider.num_conditions == 1
    loaded.pipeline.update_components(guider=ClassifierFreeGuidance(guidance_scale=4))
    assert loaded.pipeline.guider.num_conditions == 2

    # Exercise the native modular loop dispatcher around a tiny leaf block.
    seen = []

    class Leaf:
        def __call__(self, components, state, **kwargs):
            state.value += 1
            return components, state

    loaded.progress.sub_blocks = {"test_step": Leaf()}
    loaded.progress.on_step = seen.append
    state = BlockState(value=0)
    for i in range(3):
        _, state = loaded.progress.loop_step(loaded.pipeline, state, i=i, t=0)
    assert state.value == 3 and seen == [0, 1, 2]

    def cancel(_):
        raise GenerationCancelled

    loaded.progress.on_step = cancel
    try:
        loaded.progress.loop_step(loaded.pipeline, state, i=3, t=0)
    except GenerationCancelled:
        pass
    else:
        raise AssertionError("cancellation did not propagate")
    assert not torch.cuda.is_initialized()
    print("28/40-layer conversion, strict assignment, native CFG and step/cancel: passed on CPU")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
