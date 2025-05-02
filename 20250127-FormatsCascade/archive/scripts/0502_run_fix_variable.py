import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    tests = []
    for element_family, compressor, args in [
        ("int", "optimal", {}),
        ("t", None, {}),
        ("lloyd_max", None, {}),
    ]:
        for block_shape in [(None, None), (1, 64)]:
            for element_bits in [3, 3.5, 4, 4.5, 5]:
                fmt = F.Scaled(
                    element_bits=element_bits,
                    element_family=element_family,
                    scale_format=Q.BFLOAT16,
                    block_shape=block_shape,
                    scaling="absmax",
                    compressor=compressor,
                    args=args,
                )
                for kind in [
                    ET.QuantiseFixed,
                    ET.QuantiseVariable,
                    ET.QuantiseHeuristic,
                ]:
                    tests.append(kind(fmt))

    ET.run_sweep(
        [
            ET.Run("20250502-fix-variable-width", test, model, batch_size=2)
            for model in [
                "meta-llama/Llama-3.2-1B",
                "Qwen/Qwen2.5-1.5B",
                "meta-llama/Llama-3.2-3B",
                "google/gemma-3-1b-pt",
            ]
            for test in tests
        ]
    )
