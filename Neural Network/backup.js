
var predictedData = [];

console.log('Hello TensorFlow');
/**
 * Get the weather x yield data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
 async function getData() {
    const weatherDataResponse = await fetch('https://cgbrem.github.io/weather-yield-json/weather_corn_data.json');  
    const weatherData = await weatherDataResponse.json();  
    const cleaned = weatherData.map(weather => ({
      yield: weather.Yield,
      precipitation: weather.pr_rcp45,
      temp_max: weather.tasmax_rcp45,
      temp_min: weather.tasmin_rcp45,
      solar: weather.rsds_rcp45
    })).filter(d => (d.yield != null));
    console.log(cleaned);
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
    const model = tf.sequential(); 
    model.add(tf.layers.simpleRNN({units: 32, inputShape: [4]}));    
    model.add(tf.layers.dense({units: 1}));
    return model;
  }

  function normalize(num, min, max) //converts values to the range between values 0 and 1;
  {
    return (num - min) * (1/(max - min));
  }
  function denormalize(num, min, max) //reconverts values from range between values 0 and 1 to range between Min and Max;
  {
    return (num / (1/(max - min))) + min;
  }

  /**
 * Convert the input data to tensors that we can use
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any 
    // intermediate tensors.
    
    return tf.tidy(() => {
      // Step 1. Shuffle the data   
      //tf.util.shuffle(data);
  
      // Step 2. Convert data to Tensor
      // normalize inputs
      const weather = data.map(d => ({
        precipitation: d.precipitation,
        temp_max: d.temp_max,
        temp_min: d.temp_min,
        solar: d.solar
      }));
      console.log(weather);
      df = new dfd.DataFrame(weather);
      console.log(df);
      const MIN = 15.9;
      const MAX = 323.96;
      var normalizedInputs = [];
      var line = [];
      var column = 0;
      for(let x = 0; x < weather.length; x++) {// loops through each row
        line = weather[x];
        for(let y = 0 ; y < 4; y++){
          if(y == 0)
            column = line.precipitation;
          else if(y == 1)
            column = line.temp_max
          else if(y == 2)
            column = line.temp_min
          else
            column = line.solar
          ans = normalize(column, MIN, MAX);
          normalizedInputs.push(ans);
        }
      }
      console.log(normalizedInputs);

      // normalize output
      const labels = data.map(d => d.yield);
      console.log(labels);
      const LABEL_MAX = 185.9;
      const LABEL_MIN = 112.4;
      var normalizedOutputs = [];
      for(let x = 0; x < labels.length; x++) {// loops through each row
        ans = normalize(labels[x], LABEL_MIN, LABEL_MAX);
        normalizedOutputs.push(ans);
      }
      console.log(normalizedOutputs);

      // now convert each array data to a 2d tensor
      // ([num_examples, num_features_per_example])
      const inputTensor = tf.tensor2d(normalizedInputs, [104, 4]);
      console.log(inputTensor);
      const labelTensor = tf.tensor2d(normalizedOutputs, [104, 1]);
      console.log(labelTensor);

      console.log('max and mins of tensors');
      return {
        inputs: inputTensor,
        labels: labelTensor,
        MAX,
        MIN,
        LABEL_MAX,
        LABEL_MIN,
      }
    });  
  }

  async function trainModel(model, inputs, labels) {
    // optimizer: algo that governs the updates to the model as it sees examples
    // loss: tells the model how well it's doing on learning each of the batches
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError
    });
    console.log('after model compile');
    console.log(inputs)
    
    // batchSize: size of the data subsets
    const batchSize = 8;
    // epochs: number of times the model is going to look at the entire dataset
    const epochs = 50;
    
    // Start the Train Loop
    return await model.fit(inputs, labels, {
      batchSize,
      epochs, callbacks: {
        onEpochEnd: async (epoch, log) => {
          callback(epoch, log);
        }
    }});
  }
  

  function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
    // We un-normalize the data by doing the inverse of the min-max scaling 
    const [xs, preds] = tf.tidy(() => {
      // generates 100 new examples to feed the model
      const xs = tf.linspace(0, 1, 100);     
      // model.predict is how we feed the examples into the model 
      const preds = model.predict(xs.reshape([100, 1]));      
      
      // Un-normalize the data
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
    predictedData = predictedPoints;
  }

  async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    
    // tfvis.render.scatterplot(
    //   {name: 'Precipitation v Yield'},
    //   {values}, 
    //   {
    //     xLabel: 'Precipitation',
    //     yLabel: 'Yield',
    //     height: 300
    //   }
    // );
  
    // Create the model
    const model = createModel();  
    console.log('got to after model');
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    console.log('after tensor data');
    const {inputs, labels} = tensorData;
    
    // Train the model  
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    // Make some predictions using the model and compare them to the original data
    testModel(model, data, tensorData);

    // Create scatter plot of predicted values
    createPlot();
  }
  
  document.addEventListener('DOMContentLoaded', run);

  function createPlot(){
    console.log(predictedData);
    var trace1 = {
    x: [],
    y: [],
    mode: 'markers',
    type: 'scatter',
    name: 'Predicted Corn Yields',
    marker: { size: 5 }
    };

    var data = predictedData;
    data.forEach(function(val){
        trace1.x.push(val["x"]);
        trace1.y.push(val["y"]);
    });
    console.log(trace1);
    console.log(data);

    var layout = {
        autosize: false,
        width: 500,
        height: 500,
        margin: {
            l: 50,
            r: 50,
            b: 100,
            t: 100,
            pad: 4
          },
          paper_bgcolor: '#2980B9'
    };

    Plotly.newPlot('cornYieldPlot', [trace1], layout);
  }
