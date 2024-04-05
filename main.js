// free drawing canvas from Fabric.js
const fabricCanvas = new fabric.Canvas("draw", { isDrawingMode: true });
fabricCanvas.freeDrawingBrush.width = 20;

document.getElementById("clear").addEventListener("click", () => {
  fabricCanvas.clear();
});

const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");

const session = await ort.InferenceSession.create("./model.onnx");
const floatArr = new Float32Array(28 * 28);

document.getElementById("predict").addEventListener("click", async () => {
  const canvasData = ctx.getImageData(0, 0, 280, 280).data;

  // average 10x10 blocks to downscale input to 28x28
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      let sum = 0;
      for (let dy = 0; dy < 10; dy++) {
        for (let dx = 0; dx < 10; dx++) {
          const px = x * 10 + dx;
          const py = y * 10 + dy;
          const index = (py * 280 + px) * 4 + 3; // alpha channel
          sum += canvasData[index];
        }
      }
      const average = sum / 100;
      floatArr[y * 28 + x] = average / 255; // normalize to 0-1
    }
  }

  // inference
  const inputTensor = new ort.Tensor("float32", floatArr, [1, 28, 28]);
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  console.log(output);
});
