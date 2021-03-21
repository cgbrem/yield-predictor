// only need to calculate means of all years once
// otherwise it would increase the runtime

df = d3.csv("./shortwave_in_air.csv")
.then(function(data) {
    console.log(data);
    means = calculateMean(data);
    console.log(means);
})
.catch(function(error) {
    console.log(error);
})

/**
 * This method calculates the mean of a given weather feature over all the years.
 */
calculateMean = function(data){
    means = []; // stores all the yearly means
    total = 0;
    temp_count = 0;
    count = 365;
    current_year = '1979';
    index = 0;
    time = '';
    // NEED TO ACCOUNT FOR LEAP YEARS
    while(time != '2017'){
        time = data[index].TIMESTEP.split('-', 1);
        console.log("Before time: "+time)
        if(temp_count == 365){
            console.log('Total: '+total);
            means.push(total/count);
            total = 0; // resets total value
            temp_count = 0;
            index++; // goes to next row
            console.log("Count: "+temp_count+".");
        }
        else{
            total += parseInt(data[index].MEAN);
            index++; // goes to next row
            temp_count++;
            console.log("Here:"+index);
        }
    }
    return means;
}  
console.log(df);