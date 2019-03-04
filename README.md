# Pokemon Classifier

- Pokémon Classifier built with TensorFlow.js - Reognizes 50 Pokémon with 85% accuracy!

![Demo GIF](demo.gif)

##### Setup Dependencies

- run `npm install`

##### Download the MobileNet Model:

- make sure, `jq`, `gnu-sed`, `parallel` are installed - otherwise, install them with brew
- run the following:

```
mkdir models
mkdir models/mobilenet
cd models/mobilenet
curl -o model.json https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json
cat model.json | jq -r ".weightsManifest[].paths[0]" | sed 's/^/https:\/\/storage.googleapis.com\/tfjs-models\/tfjs\/mobilenet_v1_0.25_224\//' |  parallel curl -O
```

##### Add data

- create folders `./data/pokemon/test` & `./data/pokemon/train` and add subfolders with samples images for each class

##### Configure Pipeline

- open `buildModel.js` and add folder names of classes that should loaded into the model
- set hyperparameters:
  - for ~10 classes: Learning Rate = 0.0003, Epochs = 20, Batch Size = 32, Dense Units = 100
  - for ~50 classes: Learning Rate = 0.0005, Epochs = 20, Batch Size = 16, Dense Units = 500

##### Run Project

- `node run buildModel.js` to run train, evaluate and save model
- `node run buildModel.js path/to/an/image` to load saved model and predict class of image
