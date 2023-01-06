const ort = require('onnxruntime-web');
const session = ort.InferenceSession.create('/models/best.onnx', { executionProviders: ['webgl'] });
const results = session.run(feeds);

