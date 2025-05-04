import block_formats.experiments as E
import block_formats.experiments.token_prediction as ET
import block_formats.fit as F
import block_formats.quantisation as Q

if __name__ == "__main__":
    tests = []
    for element_bits in [3, 3.5, 4, 4.5, 5]:
        for element_family, compressor, args in [
            ("int", "optimal", {}),
            ("int", None, {}),
            ("fp", None, dict(exponent_bits=2)),
            ("normal", None, {}),
            ("laplace", None, {}),
            ("t", None, {}),
            ("t", None, dict(df=30)),
            ("lloyd_max", None, {}),
        ]:
            if element_family == "fp" and round(element_bits) != element_bits:
                continue  # "fp" doesn't support fractional bits

            for mode_args in (
                [dict(mode="symmetric"), dict(mode="asymmetric")]
                if element_family in ["normal", "laplace", "t"]
                or (element_family, compressor) == ("int", None)
                else [{}]
            ):
                for scaling in ["rms", "absmax"]:
                    for block_shape in [(None, None), (1, None)] + [
                        (1, b) for b in [64, 128]
                    ]:
                        for sparse_ratio in [0, 2**-7, 2**-10]:
                            fmt = F.Scaled(
                                element_bits=element_bits,
                                element_family=element_family,
                                scale_format=Q.BFLOAT16,
                                block_shape=block_shape,
                                scaling=scaling,
                                sparse_format=Q.BFLOAT16,
                                sparse_ratio=sparse_ratio,
                                compressor=compressor,
                                args=dict(**args, **mode_args),
                            )
                            tests.append(ET.QuantiseFixed(fmt))

    ET.run_sweep(
        [ET.Run("20250504-sweep", test, "meta-llama/Llama-3.1-8B") for test in tests]
    )
