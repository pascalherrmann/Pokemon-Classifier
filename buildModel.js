const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const jpeg = require("jpeg-js");

//
// Parameters
//
// for 10 classes: use 100 / 0.0003 / 20 / 32
// for 50 classes: use 500 / 0.0005 / 20 / 16
const DENSE_UNITS = 100;
const LEARNING_RATE = 0.0003;
const EPOCHS = 20;
const BATCH_SIZE = 16;

//
// Data
//
const TRAINING_DIRECTORY = "./data/pokemon/train/";
const TEST_DIRECTORY = "./data/pokemon/test/";
const CLASSES = [
	"Venusaur",
	"Blastoise",
	"Charizard",
	"Abra",
	"Kadabra",
	"Alakazam",
	"Pikachu",
	"Articuno",
	"Zapdos",
	"Moltres"
];
const DATA_LIMIT = 99999;

//
// Base Model
//
const MODEL_PATH = "./models/mobilenet/model.json";
const IMAGE_SIZE = 224;
const BASE_LAYER = "conv_pw_13_relu";
const BASE_OUTPUTSHAPE = [7, 7, 256];
const NUM_CHANNELS = 3;

// further
const CONFIG_PATH = "./pokemonlogs.json";

// Vars
var xs;
var ys;
var mobilenet;
const model = tf.sequential();

// Helpers
const jpgEnding = /\.(jpg|jpeg)$/;

// decodes img in obj {height, width, data}, where data is a flat array with RGBA for every pixel
const readImage = path => {
	const buf = fs.readFileSync(path);
	const image = jpeg.decode(buf, true);
	return image;
};

// takes an img obj and returns a flat array considering n channels (used to ommit alpha channel)
const imageByteArray = (image, numChannels) => {
	const pixels = image.data;
	const numPixels = image.width * image.height;
	const values = new Int32Array(numPixels * numChannels); //return them as flat int array [pixel1 rgb, pixel2rgb, ...)]

	for (let i = 0; i < numPixels; i++) {
		for (let channel = 0; channel < numChannels; ++channel) {
			const val = pixels[i * 4 + channel]; // 4 for rgba. But we pass only 3 as numChannels to skip alpha!
			values[i * numChannels + channel] = val;
		}
	}

	return values;
};

const readInImageTensor = path => {
	const image = readImage(path);
	const pixelArray = imageByteArray(image, NUM_CHANNELS);
	const outShape = [image.height, image.width, NUM_CHANNELS]; // to convert flat array to multidimensional array
	const tensor = tf.tensor3d(pixelArray, outShape, "int32"); // [28,28,3]
	return tensor;
};

const cropImageToSquare = imgTensor => {
	const size = Math.min(imgTensor.shape[0], imgTensor.shape[1]);
	const centerHeight = imgTensor.shape[0] / 2;
	const beginHeight = centerHeight - size / 2;
	const centerWidth = imgTensor.shape[1] / 2;
	const beginWidth = centerWidth - size / 2;
	const cropped = imgTensor.slice(
		[beginHeight, beginWidth, 0],
		[size, size, 3]
	);
	return cropped;
};

const resizeImage = (img, targetSize) => {
	const alignCorners = true;
	const resized = tf.image.resizeBilinear(
		img,
		[targetSize, targetSize],
		alignCorners
	);

	return resized;
};

const convertImageToMobileNetFormat = imageTensor => {
	return tf.tidy(() => {
		// for clean up
		// Crop the image so we're using the center square of the rectangular
		// webcam.
		const croppedImage = cropImageToSquare(imageTensor);

		const resized = resizeImage(croppedImage, IMAGE_SIZE);

		const normalized = resized
			.toFloat()
			.div(tf.scalar(127.5))
			.sub(tf.scalar(1));

		// Expand the outer most dimension so we have a batch size of 1.
		const batchedImage = normalized.expandDims(0);

		// Normalize the image between -1 and 1. The image comes in between 0-255,
		// so we divide by 127 and subtract 1.
		return batchedImage;
	});
};

const loadMobilenet = async () => {
	const mobilenet = await tf.loadModel("file://" + MODEL_PATH);

	// Return a model that outputs an internal activation.
	const layer = mobilenet.getLayer(BASE_LAYER);
	return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
};

const addExample = (example, labelNumber) => {
	const y = tf.tidy(() =>
		tf.oneHot(tf.tensor1d([labelNumber], "int32"), CLASSES.length)
	);

	if (xs == null) {
		xs = tf.keep(example);
		ys = tf.keep(y);
	} else {
		const oldX = xs;
		xs = tf.keep(oldX.concat(example, 0));

		const oldY = ys;
		ys = tf.keep(oldY.concat(y, 0));

		oldX.dispose();
		oldY.dispose();
		y.dispose();

		console.log(xs.shape[0]); // printing number of samples
	}
};

const buildModel = () => {
	const optimizer = tf.train.adam(LEARNING_RATE);

	model.add(
		// Flattens the input to a vector so we can use it in a dense layer. While
		// technically a layer, this only performs a reshape (and has no training
		// parameters).
		tf.layers.flatten({ inputShape: BASE_OUTPUTSHAPE })
	);

	model.add(
		tf.layers.dense({
			units: DENSE_UNITS,
			activation: "relu",
			kernelInitializer: "varianceScaling",
			useBias: true
		})
	);

	model.add(
		// The number of units of the last layer should correspond
		// to the number of classes we want to predict.
		tf.layers.dense({
			units: CLASSES.length,
			kernelInitializer: "varianceScaling",
			useBias: false,
			activation: "softmax"
		})
	);

	model.compile({
		optimizer: optimizer,
		loss: "categoricalCrossentropy",
		metrics: ["accuracy"]
	});
};

const loopThroughSamplesInClasses = (path, handler) => {
	for (let labelIndex = 0; labelIndex < CLASSES.length; labelIndex++) {
		fs.readdirSync(path + CLASSES[labelIndex]).forEach((file, fileIndex) => {
			handler(labelIndex, path + CLASSES[labelIndex] + "/" + file, fileIndex);
		});
	}
};

const loadData = () => {
	loopThroughSamplesInClasses(
		TRAINING_DIRECTORY,
		(labelIndex, path, sampleIndex) => {
			if (sampleIndex >= DATA_LIMIT || !jpgEnding.test(path)) return; //skip this closure
			console.log(path);
			try {
				tf.tidy(() => {
					const imageTensor = readInImageTensor(path);
					const preprocessed = convertImageToMobileNetFormat(imageTensor);
					const features = mobilenet.predict(preprocessed);
					addExample(features, labelIndex);
				});
			} catch (e) {
				console.log(e);
			}
		}
	);
};

const fit = () => {
	return new Promise((resolve, reject) => {
		model
			.fit(xs, ys, {
				epochs: EPOCHS,
				batchSize: BATCH_SIZE,
				shuffle: true
			})
			.then(history => {
				console.log(history);
				resolve(history);
			})
			.catch(e => {
				console.log(e);
				reject(e);
			});
	});
};

const predict = () => {
	var preds = 0;
	var correct = 0;

	loopThroughSamplesInClasses(
		TEST_DIRECTORY,
		(labelIndex, path, sampleIndex) => {
			if (sampleIndex >= DATA_LIMIT || !jpgEnding.test(path)) return; //skip this closure

			try {
				console.log("\n\n" + path);

				const imageTensor = readInImageTensor(path);
				const preprocessed = convertImageToMobileNetFormat(imageTensor);
				const features = mobilenet.predict(preprocessed);
				const result = model.predict(features);
				result.print();
				const predictions = Array.from(result.argMax(1).dataSync());
				console.log(
					"Prediction: " +
						CLASSES[predictions[0]] +
						(predictions[0] == labelIndex ? "✅" : "❌")
				);
				preds++;
				if (predictions[0] == labelIndex) {
					correct++;
				}
			} catch (e) {
				console.log(e);
			}
		}
	);

	const acc = ((correct / preds) * 100).toFixed(2);

	const report = {
		time: new Date(),
		params: { DENSE_UNITS, LEARNING_RATE, EPOCHS, BATCH_SIZE },
		result: { Test_Accuracy: acc },
		meta: { Classes: CLASSES.length, dataLimit: DATA_LIMIT }
	};

	console.log(
		`\n\n\nCorrect: ${correct}/${preds} = ${acc}% (${CLASSES.length} classes)`
	);

	return report;
};

const main = async () => {
	mobilenet = await loadMobilenet();
	buildModel();
	const loadingStartTime = new Date();
	loadData();
	const loadingDataTime = Math.round((new Date() - loadingStartTime) / 1000);
	const fitStartTime = new Date();
	const history = await fit();
	const fitTime = Math.round((new Date() - fitStartTime) / 1000);
	const report = predict();
	report.duration = { Training: fitTime, Loading: loadingDataTime };
	report.result.Training_Accuracy =
		history.history.acc[history.history.acc.length - 1];
	report.result.Trainings_Loss =
		history.history.loss[history.history.loss.length - 1];
	console.log(report);

	if (parseFloat(report.result.Test_Accuracy) > 85) {
		await model.save("file://./models/pokemon");
		console.log("model saved.");
	} else {
		console.log(`acc is only ${report.result.Test_Accuracy}`);
	}

	const currentLogs = readObjectFromFile();
	currentLogs.logs.push(report);
	saveObjectToFile(currentLogs);
};

const singlePrediction = async path => {
	mobilenet = await loadMobilenet();
	const myModel = await tf.loadModel("file://" + "./models/pokemon/model.json");
	const imageTensor = readInImageTensor(path);
	const preprocessed = convertImageToMobileNetFormat(imageTensor);
	const features = mobilenet.predict(preprocessed);
	const myResult = myModel.predict(features);
	const predictions = Array.from(myResult.argMax(1).dataSync());
	console.log("Prediction: " + CLASSES[predictions[0]]);
	const values = await myResult.data();
	console.log(findBestMatches(values));
};

const findBestMatches = values => {
	//values are result.data from model.predict
	const valuesAndIndices = [];
	for (let i = 0; i < values.length; i++) {
		valuesAndIndices.push({ class: CLASSES[i], value: values[i] });
	}
	valuesAndIndices.sort((a, b) => {
		return b.value - a.value;
	});

	const bestMatches = [];
	for (let i = 0; i < valuesAndIndices.length; i++) {
		if (valuesAndIndices[i].value > 0.1) {
			bestMatches.push(valuesAndIndices[i]);
		}
	}
	return bestMatches;
};

function saveObjectToFile(obj) {
	var json = JSON.stringify(obj);
	fs.writeFileSync(CONFIG_PATH, json);
}

function readObjectFromFile() {
	let rawdata = fs.readFileSync(CONFIG_PATH);
	let obj = JSON.parse(rawdata);
	return obj;
}

if (process.argv[2] == null) {
	main();
} else {
	singlePrediction(process.argv[2]);
}
