// predicting continous numbers -> regression task
// supervised learning -> predicting

// Given "precipitation" for a county, predict "corn yield"
// STEPS:
// - load the data and prepare for training
// - define the architecture of the model
// - train the model and monitor its performance as it trains
// - evaluate the trained model by making some predictions

console.log('Hello TensorFlow');
/**
 * Get the weather x yield data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
 async function getData() {
    const weatherDataResponse = await fetch('https://cgbrem.github.io/weather-yield-json/altered_corn_yields.json');  
    const weatherData = await weatherDataResponse.json();  
    const cleaned = weatherData.map(weather => ({
      yield: weather.Value,
      precipitation: weather.Precipitation,
    }))
    .filter(weather => (weather.yield != null && weather.precipitation != null));
    
    return cleaned;
  }

  // ML models are algos that take input and produce output
  // when using neural nets the algo is a set of layers of neurons
  // with weights governing their output
  // the training process learns the ideal values for those weights
  /**
   * Create a model. Basically, which functions will the model run when executing,
   * and what algorithm will our model use to compute its answers
   */
   function createModel() {
    // Create a sequential model
    // instantiates a tf.Model object
    // sequential bc its inputs flow straight down to its output
    const model = tf.sequential(); 
    
    // Add a single input layer
    // dense layer: multiplies its inputs by a matrix (called weights)
    // then adds a number (called the bias) to the result
    // This is the first layer so we need to define our inputShape
    // Units sets how big the weight matrix will be in the layer
    // units: 1 means there will be 1 weight for each of the input features of the data
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    
    // Add an output layer
    // Units is 1 bc we want to output 1 number
    model.add(tf.layers.dense({units: 1, useBias: true}));
  
    return model;
  }

  /**
 * Convert the input data to tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * Yield on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any 
    // intermediate tensors.
    
    return tf.tidy(() => {
      // Step 1. Shuffle the data   
      // this will randomize the order of the examples fed to the training algo
      // important bc during training the data is broken up into smaller subsets called batches 
      tf.util.shuffle(data);
  
      // Step 2. Convert data to Tensor
      // array for our input (precipitation)
      const inputs = data.map(d => d.precipitation)
      // array for our true output (yield), called labels in ML
      const labels = data.map(d => d.yield);
      // now convert each array data to a 2d tensor
      // has shape: [inputs.length, 1] ([num_examples, num_features_per_example])
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      // this is important bc the internals of many ML models are designed
      // to work with small numbers
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();  
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });  
  }

  async function trainModel(model, inputs, labels) {
    // Prepare the model for training.  
    // before we train, we need to compile
    // optimizer: algo that governs the updates to the model as it sees examples
    // loss: tells the model how well it's doing on learning each of the batches
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
    
    // batchSize: size of the data subsets
    const batchSize = 32;
    // epochs: number of times the model is going to look
    // at the entire dataset
    const epochs = 50;
    
    // Start the Train Loop
    // model.fit starts the training
    // tfvis.show.fitCallbacks: generate funcs that plot charts for loss and mse
    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'], 
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });
  }

  function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
    
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling 
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
      // generates 100 new examples to feed the model
      const xs = tf.linspace(0, 1, 100);     
      // model.predict is how we feed the examples into the model 
      const preds = model.predict(xs.reshape([100, 1]));      
      
      // Un-normalize the data
      // invert the operations done when normalizing
      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
      
      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
      
      // dataSync() gets a typedarray of the values stored in the tensors
      // allows us to process those values in regular JS
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
    
   
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
    
    const originalPoints = inputData.map(d => ({
      x: d.precipitation, y: d.yield,
    }));
    
    
    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data'}, 
      {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
      {
        xLabel: 'Precipitation',
        yLabel: 'Yield',
        height: 300
      }
    );

    console.log(predictedPoints);
  }

  async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
      x: d.precipitation,
      y: d.yield,
    }));
  
    tfvis.render.scatterplot(
      {name: 'Precipitation v Yield'},
      {values}, 
      {
        xLabel: 'Precipitation',
        yLabel: 'Yield',
        height: 300
      }
    );
  
    // Create the model: this will create an instance of the model and
    // show a summary of the layers on the webpage
    const model = createModel();  
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;
    
    // Train the model  
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, data, tensorData);
  }
  
  document.addEventListener('DOMContentLoaded', run);
