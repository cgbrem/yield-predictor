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
    var con = $.getJSON('./altered_corn_yields', function(json) {
        console.log(json); // this will show the info it in firebug console
    });
    const weatherData = require('./altered_corn_yields.json');
    const cleaned = weatherData.map(weather => ({
      yield: weather.Value,
      precipitation: weather.Precipitation,
    }))
    .filter(weather => (weather.yield != null && weather.precipitation != null));
    
    return cleaned;
  }

  async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
      x: d.yield,
      y: d.precipitation,
    }));
  
    tfvis.render.scatterplot(
      {name: 'Yield v Precipitation'},
      {values}, 
      {
        xLabel: 'Yield',
        yLabel: 'Precipitation',
        height: 300
      }
    );
  
    // More code will be added below
  }
  
  document.addEventListener('DOMContentLoaded', run);
