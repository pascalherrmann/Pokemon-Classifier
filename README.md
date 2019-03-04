# Pokemon Classifier

- built with TensorFlow.js
- test accuracy: 85%
- recognizes 50 Pokémon!

![Demo GIF](demo.gif)

##### Setup dependencies

- run `npm install`

##### To download the MobileNet model:

- make sure, jq, gnu-sed, parallel are installed - otherwise, install them with brew
- run the following:

```
mkdir models
mkdir models/mobilenet
cd models/mobilenet
curl -o model.json https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json
cat model.json | jq -r ".weightsManifest[].paths[0]" | sed 's/^/https:\/\/storage.googleapis.com\/tfjs-models\/tfjs\/mobilenet_v1_0.25_224\//' |  parallel curl -O
```

##### Add data

- create folders ./data/pokemon/test & ./data/pokemon/train and add subfolders with samples images for each class

##### Configure Pipeline

- add class names to buildModel.js
- set hyperparameters
