// TENSORS
// tensors are n dimensional arrays that are consumed by operators
// represent the building block for any deep learning app
// the following creates a scalar tensor
const tensor = tf.scalar(2);

// we can also convert arrays to tensors
// the following creates a tensor of the array [2,2]
const input = tf.tensor([2,2]);

// we can use input.shape to retrieve the size of the tensor
// this has shape [2]
const tensor_s = tf.tensor([2,2]).shape;

// we can also create a tensor with specific size
// below we'll create a tensor of zeros with shape [2,2]
const input = tf.zeros([2,2]);

// OPERATORS
// in order to use tensors we need to create operations on them
// below: square of a tensor
// result: [1,4,9]
const a = tf.tensor([1,2,3]);
a.square().print();

// Chaining operations
// [1,16,81]
const x = tf.tensor([1,2,3]);
const x2 = x.square().square();

// TENSOR DISPOSAL
// We generate lots of intermediate tensors
// after evaluating x2 we don't need the value of x
x.dispose();

// can no longer use the tensor 'x' in later operations
// inconvenient to do that for every tensor sooo
// use tidy()
function f(x) {
    return tf.tidy(()=>{
        const y = x.square();
        const z = x.mul(y);
        return z
    });
}

// OPTIMIZATION PROBLEM
// given a function f(x) we need to evaluate x = a that minimizes f(x)
// optimizer is an algo to minimize a function by following the gradient
// below defines the func f(x)=x^6+2x^4+3x^2+x+1
function f(x) 
{
  const f1 = x.pow(tf.scalar(6, 'int32')) //x^6
  const f2 = x.pow(tf.scalar(4, 'int32')).mul(tf.scalar(2)) //2x^4
  const f3 = x.pow(tf.scalar(2, 'int32')).mul(tf.scalar(3)) //3x^2
  const f4 = tf.scalar(1) //1
  return f1.add(f2).add(f3).add(x).add(f4)
}

// now we can iteratively minimize the func to find the value of the min
// start with value of a=2, the learning rate defines how fast we jump
// to reach the min, below we use an Adam optimizer
function minimize(epochs , lr)
{
  let y = tf.variable(tf.scalar(2)) //initial value 
  const optim = tf.train.adam(lr);  //gadient descent algorithm 
  for(let i = 0 ; i < epochs ; i++) //start minimiziation 
    optim.minimize(() => f(y));
  return y 
}
// Using a learning rate with value 0.9 we find the value of the minimum 
// after 200 iterations to be -0.16092407703399658.

// SIMPLE NEURAL NETWORK
// we will create a neural network to learn XOR (a nonlinear operation)
// we first create the training set which takes two inputs and one output