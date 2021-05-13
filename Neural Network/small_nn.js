var predictedData = [];
var acreAmount = undefined;
var url = 'https://cgbrem.github.io/weather-yield-json/weather_strawberry_data.json';
var crop = 'strawberry';
var plotName = 'yieldPlot';
var loader = 'loaderVisible';
var loaderText = 'loaderText';

console.log('Hello TensorFlow');
/**
 * Get the weather x yield data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
 async function getData(url) {
    const weatherDataResponse = await fetch(url);  
    const weatherData = await weatherDataResponse.json();  
    const cleaned = weatherData.map(weather => ({
      yield: weather.Yield,
      precipitation: weather.pr_rcp45
    })).filter(d => (d.yield != null));
    console.log(cleaned);
    return cleaned;
  }

  async function getRCP85Data(url) {
    const weatherDataResponse = await fetch(url);  
    const weatherData = await weatherDataResponse.json();  
    const cleaned = weatherData.map(weather => ({
      yield: weather.Yield,
      precipitation: weather.pr_rcp85
    })).filter(d => (d.yield != null));
    console.log(cleaned);
    return cleaned;
  }

  async function getPredictData(url) {
    const weatherDataResponse = await fetch(url);  
    const weatherData = await weatherDataResponse.json();  
    const cleaned = weatherData.map(weather => ({
      yield: weather.Yield,
      precipitation: weather.pr_rcp45
    })).filter(d => (d.yield == null));
    console.log(cleaned);
    return cleaned;
  }

  async function getPredictData_85(url) {
    const weatherDataResponse = await fetch(url);  
    const weatherData = await weatherDataResponse.json();  
    const cleaned = weatherData.map(weather => ({
      yield: weather.Yield,
      precipitation: weather.pr_rcp85
    })).filter(d => (d.yield == null));
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
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));    
    model.add(tf.layers.dense({units: 1, useBias: true}));
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
      const inputs = data.map(d => d.precipitation)
      const labels = data.map(d => d.yield);
      // has shape: [inputs.length, 1] ([num_examples, num_features_per_example])
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();  
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
      console.log('normalized');
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
    // optimizer: algo that governs the updates to the model as it sees examples
    // loss: tells the model how well it's doing on learning each of the batches
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
    
    // batchSize: size of the data subsets
    const batchSize = 8;
    // epochs: number of times the model is going to look
    // at the entire dataset
    const epochs = 50;
    
    // Start the Train Loop
    // model.fit starts the training
    // tfvis.show.fitCallbacks: generate funcs that plot charts for loss and mse
    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true
      // callbacks: tfvis.show.fitCallbacks(
      //   { name: 'Training Performance' },
      //   ['loss', 'mse'], 
      //   { height: 200, callbacks: ['onEpochEnd'] }
      // )
    });
  }
  

  async function testModel(model, inputData, normalizationData, is85) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
    var predictData = 0;  
    if(is85 == 1)
      predictData = await getPredictData_85(url);
    else
      predictData = await getPredictData(url);
    const inputs = predictData.map(d => d.precipitation)
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const inputMax2 = inputTensor.max();
    const inputMin2 = inputTensor.min(); 
    const normalizedInput = inputTensor.sub(inputMin2).div(inputMax2.sub(inputMin2));


    const [xs, preds] = tf.tidy(() => {

      const xs = normalizedInput; 
      const preds = model.predict(xs);      
      
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

    predictedData = predictedPoints;
    return predictedData;
  }

  async function run() {
    if(acreAmount === undefined){
      alert("Please Enter Acre Amount");
    }
    else if(acreAmount === 0){
      alert("Please Update Acre Amount");
    }
    else{
    document.getElementById('acreAmountComp').disabled=false;
    document.getElementById('cropSelectedComp').disabled=false;
    document.getElementById('compareButton').disabled=false;
    document.getElementById(loader).style.visibility = "visible";
    document.getElementById(loaderText).style.visibility = "visible";
    document.getElementById(plotName).style.visibility = "hidden";

    // Load and plot the original input data that we are going to train on.
    console.log(url);
    const data = await getData(url);
    const data_85 = await getRCP85Data(url);
    console.log(data_85)
    const values = data.map(d => ({
      x: d.precipitation,
      y: d.yield
    }));
  
    // Create the model
    const model = createModel();  
    console.log('got to after model');
    
    // -----FOR RCP_45 DATA-----
    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    var {inputs, labels} = tensorData;
    // Train the model  
    await trainModel(model, inputs, labels);
    console.log('Done Training');
    // Make some predictions using the model and compare them to the original data
    const predicted = await testModel(model, data, tensorData, 0);

    // -----FOR RCP_85 DATA-----
    const model_85 = createModel();  
    const tensorData_85 = convertToTensor(data_85);
    console.log(tensorData_85);
    inputs = tensorData_85.inputs;
    labels = tensorData_85.labels;
    console.log(inputs);
    console.log(labels);
    // Train the model  
    await trainModel(model_85, inputs, labels);
    console.log('Done Training 85');
    // Make some predictions using the model and compare them to the original data
    const predicted_85 = await testModel(model_85, data_85, tensorData_85, 1);

    // Create scatter plot of predicted values
    createPlot(predicted, predicted_85);
  }
  
  function createPlot(predicted, predicted_85){
    // -----FOR RCP 45-----
    var count = 0;
    var total = 0;
    var means = []; // length of 11, for those years
    for(let i = 0; i < 88; i++){
      if(count == 7){
        means.push(total/count);
        total = 0;
        count = 0;
      }
      else{
        total += (predicted[i].y) * acreAmount;
        count++;
      }
    }
    console.log(means);

    // -----FOR RCP 85-----
    count = 0;
    total = 0;
    var means_85 = []; // length of 11, for those years
    for(let i = 0; i < 88; i++){
      if(count == 7){
        means_85.push(total/count);
        total = 0;
        count = 0;
      }
      else{
        total += (predicted_85[i].y) * acreAmount;
        count++;
      }
    }
    console.log(means_85);

    var trace1 = {
    x: [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031], // years 2021-2031
    y: [], // predicted yields
    mode: 'lines+markers',
    type: 'scatter',
    name: 'RCP 4.5',
    marker: { size: 5 }
    };

    means.forEach(function(val){
      trace1.y.push(val);
    });
    console.log(trace1);

    var trace2 = {
      x: [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031], // years 2021-2031
      y: [], // predicted yields
      mode: 'lines+markers',
      type: 'scatter',
      name: 'RCP 8.5',
      marker: { size: 5 }
      };
  
      means_85.forEach(function(val){
        trace2.y.push(val);
      });
      console.log(trace2);

    var cornLayout = {
      autosize: false,
      width: 850,
      height: 300,
        margin: {
            l: 50,
            r: 50,
            b: 100,
            t: 100,
            pad: 4
          },
        title: {text: 'Predicted Total Corn Yields'},
        xaxis: {title: {text: 'Years'}},
        yaxis: {title: {text: 'Bushels of Corn'}}
    };

    var strawberryLayout = {
      autosize: false,
      width: 850,
      height: 300,
      margin: {
          l: 50,
          r: 50,
          b: 50,
          t: 50,
          pad: 4
        },
      title: {text: 'Predicted Total Strawberry Yields'},
      xaxis: {title: {text: 'Years'}},
      yaxis: {title: {text: 'CWT of Strawberries'}}
    };

    var snapLayout = {
      autosize: false,
      width: 850,
      height: 300,
      margin: {
          l: 50,
          r: 50,
          b: 100,
          t: 100,
          pad: 4
        },
      title: {text: 'Predicted Total Snap Pea Yields'},
      xaxis: {title: {text: 'Years'}},
      yaxis: {title: {text: 'CWT of Snap Peas'}}
    };

    var tomatoLayout = {
      autosize: false,
      width: 850,
      height: 300,
      margin: {
          l: 50,
          r: 50,
          b: 100,
          t: 100,
          pad: 4
        },
      title: {text: 'Predicted Total Tomato Yields'},
      xaxis: {title: {text: 'Years'}},
      yaxis: {title: {text: 'CWT of Tomatoes'}}
    };

    var config = {responsive: true}

    if(crop === 'corn')
      Plotly.newPlot(plotName, [trace1, trace2], cornLayout, config);
    if(crop === 'strawberry')
      Plotly.newPlot(plotName, [trace1, trace2], strawberryLayout, config);
    if(crop === 'snap')
      Plotly.newPlot(plotName, [trace1, trace2], snapLayout, config);
    if(crop === 'tomato')
      Plotly.newPlot(plotName, [trace1, trace2], tomatoLayout, config);
    document.getElementById(loader).style.visibility = "hidden";
    document.getElementById(loaderText).style.visibility = "hidden";
    document.getElementById(plotName).style.visibility = "visible";
    crop = "strawberry";
    acreAmount = 0;
    }
  }

  function updateAcre(param){
    if(param === "model")
      selected = document.getElementById('acreAmount').value;
    else if(param === "comp")
      selected = document.getElementById('acreAmountComp').value;
    acreAmount = selected;
    console.log(acreAmount);
  }

  function updateCrop(param){
    var selected;
    if(param === "model")
      selected = document.getElementById('cropSelected').value;
    else if(param === "comp")
      selected = document.getElementById('cropSelectedComp').value;
    if(selected === 'strawberry'){
      url = 'https://cgbrem.github.io/weather-yield-json/weather_strawberry_data.json';
      crop = 'strawberry';
    }
    if(selected === 'corn'){
      url = 'https://cgbrem.github.io/weather-yield-json/weather_corn_data.json';
      crop = 'corn';
    }
    if(selected === 'tomato'){
      url = 'https://cgbrem.github.io/weather-yield-json/weather_tomato_data.json';
      crop = 'tomato';
    }
    if(selected === 'snap_peas'){
      url = 'https://cgbrem.github.io/weather-yield-json/weather_snap_data.json';
      crop = 'snap';
    }
    console.log(crop);
  }

  function updatePlot(button){
    if(button === "modelButton"){
      plotName = "yieldPlot";
      loader = "loaderVisible";
      loaderText = "loaderText";
    }
    if(button === "compareButton"){
      plotName = "comparePlot";
      loader = "loaderVisibleComp";
      loaderText = "loaderTextComp";
    }
  }
