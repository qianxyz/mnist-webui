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

// -------------------------------------------
// Table containing the possibility of classes
// -------------------------------------------

const pTable = document.getElementById("pTable");

// ------------------------
// Create inference session
// ------------------------

const session = await ort.InferenceSession.create('./model.onnx');

// ---------------------------------
// Button for running the prediction
// ---------------------------------

const predictButton = document.getElementById("predictButton");

predictButton.addEventListener('click', predict);

async function predict() {
	const arr = imageToArray();

	const inputTensor = new ort.Tensor('float32', arr, [1, imageSize, imageSize]);

	const feeds = {};
	feeds[session.inputNames[0]] = inputTensor;

	const results = await session.run(feeds);

	const output = results[session.outputNames[0]].data;

	for (let i = 0; i < 10; i += 1) {
		pTable.rows[i].cells[1].innerHTML = output[i].toFixed(2);
	}
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

// ------------------------------
// Button for clearing the canvas
// ------------------------------

const clearButton = document.getElementById("clearButton");

clearButton.addEventListener('click', () => {
	drawCtx.clearRect(0, 0, canvasSize, canvasSize);
});
