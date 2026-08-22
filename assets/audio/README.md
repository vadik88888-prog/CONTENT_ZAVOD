# Packaged Audio Intelligence assets

`yamnet.onnx` is the Apache-2.0 ONNX conversion published at
<https://huggingface.co/audiomagic/yamnet-onnx>. It is a format conversion of
Google YAMNet v1; the weights were not retrained. Upstream implementation:
<https://github.com/tensorflow/models/tree/master/research/audioset/yamnet>.

- model SHA-256: `d3835ffbbd4a1bb3e777f0ca217b5007907f5171dd5d17c4236b95b2af8f908e`
- class map SHA-256: `cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2`
- model/upstream code license: Apache-2.0 (`YAMNET_APACHE_LICENSE.txt`)
- AudioSet ontology/class labels: CC BY 4.0, Google Inc.;
  <https://research.google.com/audioset/ontology/index.html>

The application verifies both pinned hashes before inference and never
downloads a model during source analysis.
