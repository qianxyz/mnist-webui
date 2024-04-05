// free drawing canvas from Fabric.js
const fabricCanvas = new fabric.Canvas("draw", { isDrawingMode: true });
fabricCanvas.freeDrawingBrush.width = 20;

document.getElementById("clear").addEventListener("click", () => {
  fabricCanvas.clear();
});

const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");

const modelInput = new Float32Array(28 * 28);

document.getElementById("predict").addEventListener("click", () => {
  const canvasData = ctx.getImageData(0, 0, 280, 280).data;

  // downscale the image to input
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
      const average = sum / 100; // average of 10x10 block
      modelInput[y * 28 + x] = average / 255; // normalize to 0-1
    }
  }

  console.log(modelInput);
});

// // inference
// const session = await ort.InferenceSession.create("./model.onnx");
//
// const predictButton = document.getElementById("predictButton");
//
// predictButton.addEventListener("click", predict);
//
// async function predict() {
//   const arr = imageToArray();
//
//   const inputTensor = new ort.Tensor("float32", arr, [1, imageSize, imageSize]);
//
//   const feeds = {};
//   feeds[session.inputNames[0]] = inputTensor;
//
//   const results = await session.run(feeds);
//
//   const output = results[session.outputNames[0]].data;
//
//   for (let i = 0; i < 10; i += 1) {
//     pTable.rows[i].cells[1].innerHTML = output[i].toFixed(2);
//   }
// }
//
// // Convert the canvas into a Float32Array of length 28*28.
// function imageToArray() {
//   const draw = drawCtx.getImageData(0, 0, canvasSize, canvasSize).data;
//   const arr = new Float32Array(imageSize * imageSize);
//
//   for (let row = 0; row < imageSize; row += 1) {
//     for (let col = 0; col < imageSize; col += 1) {
//       let pixelSum = 0;
//
//       for (let i = 0; i < canvasScale; i += 1) {
//         for (let j = 0; j < canvasScale; j += 1) {
//           let trueRow = row * canvasScale + i;
//           let trueCol = col * canvasScale + j;
//           let pixelIndex = trueRow * canvasSize + trueCol;
//           let alpha = draw[pixelIndex * 4 + 3];
//           pixelSum += alpha;
//         }
//       }
//
//       pixelSum /= 255.0 * canvasScale * canvasScale;
//       arr[row * imageSize + col] = pixelSum;
//     }
//   }
//
//   return arr;
// }
//
// // ------------------------------
// // Button for clearing the canvas
// // ------------------------------
//
// const clearButton = document.getElementById("clearButton");
//
// clearButton.addEventListener("click", () => {
//   drawCtx.clearRect(0, 0, canvasSize, canvasSize);
// });
