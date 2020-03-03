import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

from MLBOX.Database.core.features import Feature  # noqa: E402
from MLBOX.Database.core.parsers import ParserFMT  # noqa: E402

from MLBOX.Database.core.features import _tffeature_int64  # noqa: E402
from MLBOX.Database.core.features import _tffeature_float  # noqa: E402
from MLBOX.Database.core.features import _tffeature_bytes  # noqa: E402


class TestParserBase:

    def setup_method(self):
        class IdentityParse(Feature):
            def _parse_from(self, **kwargs):
                return kwargs

        class fbytes(IdentityParse):
            encoded_features = {"fbytes": tf.io.FixedLenFeature([], tf.string)}

            def _create_from(self, fbytes_input=""):
                return {"fbytes": _tffeature_bytes(fbytes_input)}

        class fint(IdentityParse):
            encoded_features = {"fint": tf.io.FixedLenFeature([], tf.int64)}

            def _create_from(self, fint_input=0):
                return {"fint": _tffeature_int64(fint_input)}

        class ffloat(IdentityParse):
            encoded_features = {"ffloat": tf.io.FixedLenFeature([], tf.float32)}

            def _create_from(self, ffloat_input=0.0):
                return {"ffloat": _tffeature_float(ffloat_input)}

        class sample_parser(ParserFMT):
            feats = [fbytes(), fint(), ffloat()]
            @property
            def features(self): return sample_parser.feats

        self.parser = sample_parser()

    def test_to_example(self):
        """parser should convert tf.train.Feature from Feature object into tf.train.example"""
        example = self.parser.to_example(
            fbytes_input="FireBomber", fint_input=2059, ffloat_input=3.14
        )
        assert example.features.feature["fbytes"].bytes_list.value == [b"FireBomber"]
        assert example.features.feature["fint"].int64_list.value == [2059]

        float_val = example.features.feature["ffloat"].float_list.value[0]
        assert float_val == pytest.approx(3.14)

    def test_parse_example(self):
        """parser should parse tf.train.Example to tf.tensor"""
        features = {
            "fbytes": _tffeature_bytes("myleneflare"),
            "fint": _tffeature_int64(2045),
            "ffloat": _tffeature_float(2.718)
        }
        example = tf.train.Example(
            features=tf.train.Features(feature=features)
        )

        parsed = self.parser.parse_example(example)
        parsed = {k: v.numpy() for k, v in parsed.items()}
        assert parsed["fbytes"] == b"myleneflare"
        assert parsed["fint"] == 2045
        assert parsed["ffloat"] == pytest.approx(2.718)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
