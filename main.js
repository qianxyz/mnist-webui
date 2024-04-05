// draw empty chart
const colorMax = "#0d6efd";
const colorOthers = "#adb5bd";
const chart = Highcharts.chart("chart", {
  chart: { type: "column", margin: 0 },
  credits: { enabled: false },
  title: { text: undefined },
  xAxis: {
    categories: [..."0123456789"],
    labels: { reserveSpace: false, y: -10 },
  },
  yAxis: { min: 0, max: 1, visible: false },
  legend: { enabled: false },
  tooltip: { enabled: false },
  plotOptions: {
    column: {
      pointPadding: 0,
      borderWidth: 0,
      groupPadding: 0,
      shadow: false,
    },
    series: {
      colorByPoint: true,
    },
  },
  series: [
    {
      data: Array(10).fill(0),
      colors: Array(10).fill(colorOthers),
    },
  ],
});

// free drawing canvas from Fabric.js
const fabricCanvas = new fabric.Canvas("draw", { isDrawingMode: true });
fabricCanvas.freeDrawingBrush.width = 20;

document.getElementById("clear").addEventListener("click", () => {
  fabricCanvas.clear();
  // clear chart
  chart.series[0].update({
    data: Array(10).fill(0),
    colors: Array(10).fill(colorOthers),
  });
});

const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");

const session = await ort.InferenceSession.create("./model.onnx");
const floatArr = new Float32Array(28 * 28);

document.getElementById("predict").addEventListener("click", async () => {
  const canvasData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

  // average blocks to downscale input to 28x28
  const blockSize = Math.floor(canvas.width / 28);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      let sum = 0;
      for (let dy = 0; dy < blockSize; dy++) {
        for (let dx = 0; dx < blockSize; dx++) {
          const px = x * blockSize + dx;
          const py = y * blockSize + dy;
          const index = (py * canvas.width + px) * 4 + 3; // alpha channel
          sum += canvasData[index];
        }
      }
      const average = sum / (blockSize * blockSize);
      floatArr[y * 28 + x] = average / 255; // normalize to 0-1
    }
  }

  // inference
  const inputTensor = new ort.Tensor("float32", floatArr, [1, 28, 28]);
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  // draw chart
  const maxIndex = output.indexOf(Math.max(...output));
  const colors = Array(10).fill(colorOthers);
  colors[maxIndex] = colorMax;
  chart.series[0].update({
    data: Array.from(output),
    colors: colors,
  });
});
