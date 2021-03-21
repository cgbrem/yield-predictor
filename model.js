// years 2000 - 2016
// only precipitation as an added column
// RNN predicting yields

const TRAIN_MIN_ROW = 0;
const TRAIN_MAX_ROW = 514;
const VAL_MIN_ROW = 515;
const VAL_MAX_ROW = 771;

var weatherYield = 0;

df = d3.csv("./altered_corn_yields.csv")
.then(function(data) {
    console.log(data);
    weatherYield = data;
})
.catch(function(error) {
    console.log(error);
})

/**
 * Build a linear-regression model for the temperature-prediction problem.
 *
 * @param {tf.Shape} inputShape Input shape (without the batch dimenson).
 * @returns {tf.LayersModel} A TensorFlow.js tf.LayersModel instance.
 */
 function buildLinearRegressionModel(inputShape) {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape}));
    model.add(tf.layers.dense({units: 1}));
    return model;
  }

/**
 * Build a model for the temperature-prediction problem.
 *
 * @param {string} modelType Model type.
 * @param {number} numTimeSteps Number of time steps in each input.
 *   exapmle
 * @param {number} numFeatures Number of features (for each time step).
 * @returns A compiled instance of `tf.LayersModel`.
 */
 export function buildModel(modelType, numTimeSteps, numFeatures) {
    const inputShape = [numTimeSteps, numFeatures];
  
    console.log(`modelType = ${modelType}`);
    let model;
    if (modelType === 'mlp') {
      model = buildMLPModel(inputShape);
    } else if (modelType === 'mlp-l2') {
      model = buildMLPModel(inputShape, tf.regularizers.l2());
    } else if (modelType === 'linear-regression') {
      model = buildLinearRegressionModel(inputShape);
    } else if (modelType === 'mlp-dropout') {
      const regularizer = null;
      const dropoutRate = 0.25;
      model = buildMLPModel(inputShape, regularizer, dropoutRate);
    } else if (modelType === 'simpleRNN') {
      model = buildSimpleRNNModel(inputShape);
    } else if (modelType === 'gru') {
      model = buildGRUModel(inputShape);
      // TODO(cais): Add gru-dropout with recurrentDropout.
    } else {
      throw new Error(`Unsupported model type: ${modelType}`);
    }
  
    model.compile({loss: 'meanAbsoluteError', optimizer: 'rmsprop'});
    model.summary();
    return model;
  }
  
  /**
   * Train a model on the weather & yield data.
   *
   * @param {tf.LayersModel} model A compiled tf.LayersModel object. It is
   *   expected to have a 3D input shape `[numExamples, timeSteps, numFeatures].`
   *   and an output shape `[numExamples, 1]` for predicting the temperature
   * value.
   * @param {JenaWeatherData} jenaWeatherData A JenaWeatherData object.
   * @param {boolean} normalize Whether to used normalized data for training.
   * @param {boolean} includeDateTime Whether to include date and time features
   *   in training.
   * @param {number} lookBack Number of look-back time steps.
   * @param {number} step Step size used to generate the input features.
   * @param {number} delay How many steps in the future to make the prediction
   *   for.
   * @param {number} batchSize batchSize for training.
   * @param {number} epochs Number of training epochs.
   * @param {tf.Callback | tf.CustomCallbackArgs} customCallback Optional callback
   *   to invoke at the end of every epoch. Can optionally have `onBatchEnd` and
   *   `onEpochEnd` fields.
   */
  export async function trainModel(
      model, yieldWeatherData, normalize, includeDateTime, lookBack, step, delay,
      batchSize, epochs, customCallback) {
    const trainShuffle = true;
    const trainDataset =
        tf.data
            .generator(
                () => yieldWeatherData.getNextBatchFunction(
                    trainShuffle, lookBack, delay, batchSize, step, TRAIN_MIN_ROW,
                    TRAIN_MAX_ROW, normalize, includeDateTime))
            .prefetch(8);
    const evalShuffle = false;
    const valDataset = tf.data.generator(
        () => yieldWeatherData.getNextBatchFunction(
            evalShuffle, lookBack, delay, batchSize, step, VAL_MIN_ROW,
            VAL_MAX_ROW, normalize, includeDateTime));
  
    await model.fitDataset(trainDataset, {
      batchesPerEpoch: 500,
      epochs,
      callbacks: customCallback,
      validationData: valDataset
    });
  }