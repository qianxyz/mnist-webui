// MNIST image size is 28x28, the actual canvas is scaled 10x
const imageSize = 28;
const canvasScale = 10;
const canvasSize = imageSize * canvasScale;

// -----------------
// Canvas to draw on
// -----------------

const drawCanvas = document.getElementById("drawCanvas");
const drawCtx = drawCanvas.getContext("2d");

drawCanvas.width = canvasSize;
drawCanvas.height = canvasSize;

// set the drawing properties
drawCtx.lineWidth = 25;
drawCtx.lineJoin = 'round';
drawCtx.lineCap = 'round';
drawCtx.strokeStyle = '#000';

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// event hooks
function startDrawing(e) {
	isDrawing = true;
	lastX = e.offsetX;
	lastY = e.offsetY;
}

function draw(e) {
	if (!isDrawing) return;

	drawCtx.beginPath();
	drawCtx.moveTo(lastX, lastY);
	drawCtx.lineTo(e.offsetX, e.offsetY);
	drawCtx.stroke();

	lastX = e.offsetX;
	lastY = e.offsetY;
}

function stopDrawing() {
	isDrawing = false;
}

drawCanvas.addEventListener('mousedown', startDrawing);
drawCanvas.addEventListener('mousemove', draw);
drawCanvas.addEventListener('mouseup', stopDrawing);
drawCanvas.addEventListener('mouseout', stopDrawing);

// ----------------------------------
// Canvas to show the pixelated image
// ----------------------------------

const showCanvas = document.getElementById("showCanvas");
const showCtx = showCanvas.getContext("2d");

showCanvas.width = canvasSize;
showCanvas.height = canvasSize;

// ------------------------
// Create inference session
// ------------------------

const session = await ort.InferenceSession.create('./model.onnx');

// ---------------------------------
// Button for running the prediction
// ---------------------------------

const button = document.getElementById("predictButton");

button.addEventListener('click', predict);

async function predict() {
	const arr = imageToArray();

	// helper: draw the pixelated version
	for (let row = 0; row < imageSize; row += 1) {
		for (let col = 0; col < imageSize; col += 1) {
			let alpha = arr[row * imageSize + col];
			alpha = Math.floor(alpha * 255);
			showCtx.fillStyle = `rgb(${alpha}, ${alpha}, ${alpha})`;
			showCtx.fillRect(col * canvasScale, row * canvasScale, canvasScale, canvasScale);
		}
	}

	const inputTensor = new ort.Tensor('float32', arr, [1, imageSize, imageSize]);

	const feeds = {};
	feeds[session.inputNames[0]] = inputTensor;

	const results = await session.run(feeds);

	const output = results[session.outputNames[0]];

	console.log(output);
}

// Convert the canvas into a Float32Array of length 28*28.
function imageToArray() {
	const draw = drawCtx.getImageData(0, 0, canvasSize, canvasSize).data;
	const arr = new Float32Array(imageSize * imageSize);

	for (let row = 0; row < imageSize; row += 1) {
		for (let col = 0; col < imageSize; col += 1) {
			let pixelSum = 0;

			for (let i = 0; i < canvasScale; i += 1) {
				for (let j = 0; j < canvasScale; j += 1) {
					let trueRow = row * canvasScale + i;
					let trueCol = col * canvasScale + j;
					let pixelIndex = trueRow * canvasSize + trueCol;
					let alpha = draw[pixelIndex * 4 + 3];
					pixelSum += alpha;
				}
			}

			pixelSum /= 255.0 * canvasScale * canvasScale;
			arr[row * imageSize + col] = pixelSum
		}
	}

	return arr;
}
