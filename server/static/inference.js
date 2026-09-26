/* All model execution providers are explicitly WASM. This worker owns no GPU. */
let model, processor, layout, tf, ort;
let root;
const progress = (message) => self.postMessage({ progress: message });
const threads = () =>
  globalThis.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 2)
    : 1;
async function detect(call) {
  if (!ort) {
    ort = await import(root + "runtime/ort.wasm.min.mjs");
    ort.env.wasm.wasmPaths = root + "runtime/";
    ort.env.wasm.numThreads = threads();
  }
  if (!layout) {
    progress("正在加载版面模型…");
    layout = await ort.InferenceSession.create(root + "layout.onnx", {
      executionProviders: ["wasm"],
    });
  }
  const response = await fetch(call.image);
  if (!response.ok) throw new Error("无法读取页面");
  const image = await createImageBitmap(await response.blob());
  const width = image.width,
    height = image.height;
  const canvas = new OffscreenCanvas(800, 800),
    context = canvas.getContext("2d");
  context.drawImage(image, 0, 0, 800, 800);
  image.close();
  const rgba = context.getImageData(0, 0, 800, 800).data,
    pixels = new Float32Array(3 * 800 * 800);
  for (let i = 0; i < 800 * 800; i++)
    for (let c = 0; c < 3; c++)
      pixels[c * 800 * 800 + i] = rgba[4 * i + c] / 255;
  progress("正在检测小节…");
  const outputs = await layout.run({
    image: new ort.Tensor("float32", pixels, [1, 3, 800, 800]),
    im_shape: new ort.Tensor("float32", new Float32Array([800, 800]), [1, 2]),
    scale_factor: new ort.Tensor(
      "float32",
      new Float32Array([800 / height, 800 / width]),
      [1, 2],
    ),
  });
  const rows = outputs["fetch_name_0"],
    boxes = [],
    labels = [
      "measure_tab",
      "measure_notation",
      "measure_both",
      "tempo_region",
    ];
  const stride = rows.dims.at(-1);
  for (let i = 0; i < rows.data.length; i += stride) {
    const cls = Math.round(rows.data[i]),
      score = rows.data[i + 1];
    if (!labels[cls] || score < 0.25) continue;
    const coordinate = Array.from(rows.data.slice(i + 2, i + 6)).map((v, j) =>
      Math.max(0, Math.min(v, j % 2 ? height : width)),
    );
    if (coordinate[2] > coordinate[0] && coordinate[3] > coordinate[1])
      boxes.push({ label: labels[cls], score, coordinate });
  }
  for (const tensor of Object.values(outputs)) tensor.dispose();
  return { boxes };
}
async function recognize(call) {
  if (!tf) {
    tf = await import(root + "runtime/transformers.min.js");
    tf.env.allowRemoteModels = false;
    tf.env.allowLocalModels = true;
    tf.env.localModelPath = root;
    tf.env.useBrowserCache = true;
    tf.env.backends.onnx.wasm.wasmPaths = root + "runtime/";
    tf.env.backends.onnx.wasm.numThreads = threads();
  }
  if (!model) {
    progress("正在下载并加载 OCR 模型，首次使用需要较长时间…");
    model = await tf.GlmOcrForConditionalGeneration.from_pretrained(call.kind, {
      device: "wasm",
      dtype: "q8",
      progress_callback: (event) => {
        if (event.status === "progress" && event.total)
          progress(
            `加载模型 ${Math.round((event.loaded / event.total) * 100)}%`,
          );
      },
    });
    processor = await tf.Glm46VProcessor.from_pretrained(call.kind);
  }
  const images = [];
  for (const message of call.messages)
    for (const item of message.content)
      if (item.type === "image")
        images.push(await tf.RawImage.fromURL(item.url));
  const text = processor.apply_chat_template(call.messages, {
    add_generation_prompt: true,
  });
  const inputs = await processor(text, images);
  progress("正在使用浏览器 CPU 识别…");
  const generated = await model.generate({
    ...inputs,
    do_sample: false,
    max_new_tokens: call.max_new_tokens,
  });
  const tokens = generated.slice(null, [inputs.input_ids.dims.at(-1), null]);
  const output = processor.batch_decode(tokens, {
    skip_special_tokens: call.skip_special_tokens,
  })[0];
  const count = tokens.dims.at(-1);
  for (const tensor of [...Object.values(inputs), generated, tokens])
    if (tensor?.dispose) tensor.dispose();
  return { text: output, tokens: count };
}
self.onmessage = async (event) => {
  root = event.data.model_root;
  try {
    self.postMessage({
      result:
        event.data.kind === "layout"
          ? await detect(event.data)
          : await recognize(event.data),
    });
  } catch (error) {
    console.error(error);
    self.postMessage({
      result: { error: String(error.message || error).slice(0, 500) },
    });
  }
};
