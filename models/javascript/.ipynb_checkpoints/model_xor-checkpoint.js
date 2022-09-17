class Model{
    constructor(){
        // (1) code_layers_class_instance
        this.layers = [ Dense(activation=sigmoid, weights=[ new Matrix([[-0.5692449808120728, 7.034882068634033, 0.8901565670967102, -1.2982566356658936, -7.78250789642334, -0.596753716468811, -3.094672679901123, 0.4607388377189636, -0.5057770609855652, -0.3972644507884979], [-0.9048697352409363, 6.902655124664307, 1.6131205558776855, -0.23207269608974457, 5.748035907745361, -0.43503740429878235, 5.758194923400879, 2.4055724143981934, -1.5594075918197632, -1.570223331451416]]), new Matrix([-0.07483713328838348, -4.254740238189697, -0.608847975730896, -0.5387024283409119, -0.8356730341911316, -0.22868682444095612, -0.4641977548599243, -1.3353506326675415, 0.20014885067939758, 0.09103742241859436]) ]), Dense(activation=linear, weights=[ new Matrix([[0.12166593223810196], [1.637164831161499], [-0.4853067398071289], [0.09327518194913864], [1.5986828804016113], [0.031022746115922928], [-1.3279681205749512], [-0.7030462622642517], [0.2715253531932831], [0.3019183874130249]]), 0 ]) ]; 
        this.numLayers = this.layers.length;
    }
        
    pred(x){
        let output = parseInput(x);
        for ( let idx = 0, idx < this.numLayers ; idx++){
            output = this.layers[idx];
        }
        return output;
    }
}

function parseInput(x){
    if ( x instanceof Matrix ){
        return new Matrix( x.mat );
    } else if ( x instanceof Array ){
        return  x[0].length ? x : [x] ;
    } else {
        return null;
    }
}

// (2) code_apply_func
function applyFunc(matrix,fnc){
    const resMatrix = new Matrix(matrix.mat);
    for (let row = 0; row < matrix.rows; row++) {
        for (let col = 0; col < matrix.cols; col++) {
          resMatrix.mat[row][col] = fnc(matrix.mat[row][col]);
        }
    }
    return resMatrix;
}

// (3) code_act_funcs
sigmoid = (m) => applyFunc(m, (v) => (1/(1+Math.exp(-v))) );
linear = (m) => m;

// (4) code_layers_class
class Dense {
  
    constructor(activation, weights ) {
        this.activation = activation;
        [ this.kernel , this.bias ] = weights;
    }
    
    predict(x){
        return this.activation(x.dot(this.kernel).add(this.bias);
    }   
}

// (5) code_matrix
class Matrix {
  rows = null;
  cols = null;

  constructor(matrix) {
    this.rows = matrix.length;
    this.cols = matrix[0].length;
    this.mat = matrix;
    this.updateMinPad();
  }

  static Zeros(rows, cols) {
    const matrix = [];
    for (let row = 0; row < rows; row++) {
      matrix.push([...Array(cols).fill(0)])
    }
    return new Matrix(matrix);
  }

  shape(){
    return [this.rows,this.cols];
  }

  get(row = null, col = null) {
    row = (row == -1) ? this.rows - 1 : row;
    col = (col == -1) ? this.cols - 1 : col;
    if (col != null) {
      if (row != null) return this.mat[row][col];
      else {
        const column = [];
        for (let row = 0; row < this.rows; row++) {
          column.push([this.mat[row][col]]);
        }
        return new Matrix(column);
      }
    } else {
      if (row != null) return new Matrix([this.mat[row]]);
      else return new Matrix(this.mat);
    }
  }

  max() {
    let maxValue = -Infinity;
    let currValue;
    for (let row = 0; row < this.rows; row++) {
      currValue = Math.max(...this.mat[row]);
      if (currValue > maxValue) {
        maxValue = currValue;
      }
    }
    return maxValue;
  }

  min() {
    let minValue = Infinity;
    let currValue;
    for (let row = 0; row < this.rows; row++) {
      currValue = Math.min(...this.mat[row]);
      if (currValue < minValue) {
        minValue = currValue;
      }
    }
    return minValue;
  }

  round(dec=1){
    const OP = 10 ** dec;
    const newMatrix = Matrix.Zeros(this.rows, this.cols);
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        newMatrix.mat[row][col] = Math.round(this.mat[row][col]*OP)/OP;
      }
    }
    newMatrix.updateMinPad();
    return newMatrix;
  }

  updateMinPad() {
    const flatMatrix = this.mat.reduce( (acc,x) => acc.concat(x), [] );
    const idxLength = flatMatrix.map( (x,idx) => [idx, String(x).length]);
    idxLength.sort();
    this.minPad = idxLength.at(-1)[1];
  }

  print() {
    let result = "";
    const addPad = (x) => String(x).padStart(this.minPad, " ");
    for (let row = 0; row < this.rows; row++) {
      result += `[ ${this.mat[row].map(addPad).join(" , ")} ]\n`;
    }
    console.log(result);
  }

  addValue(num) {
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < this.cols; col++) {
        this.mat[row][col] += num;
      }
    }
    return this;
  }

  add(matrix) {
    if (this.cols !== matrix.cols) return null;
    if (matrix.rows === 1){
      const repMat = [];
      for(let idx=0; idx < this.rows; idx++) repMat.push([...matrix.mat[0]]);
      matrix = new Matrix(repMat);
    }
    const newMatrix = Matrix.Zeros(this.rows, matrix.cols);
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < matrix.cols; col++) {
        newMatrix.mat[row][col] = this.mat[row][col] + matrix.mat[row][col];
      }
    }
    newMatrix.updateMinPad();
    return newMatrix;
  }

  dot(matrix) {
    if (this.cols !== matrix.rows) return null;
    const REPS = this.cols;
    let value;
    const newMatrix = Matrix.Zeros(this.rows, matrix.cols);
    for (let row = 0; row < this.rows; row++) {
      for (let col = 0; col < matrix.cols; col++) {
        value = 0;
        for (let rep = 0; rep < REPS; rep++) {
          value += this.mat[row][rep] * matrix.mat[rep][col];
        }
        newMatrix.mat[row][col] = value;
      }
    }
    newMatrix.updateMinPad();
    return newMatrix;
  }
}

const matA = new Matrix([[0, 2, 3], [4, 5, 6]]);
const matB = new Matrix([[10, 2], [4, 5], [2, 1]]);
const matC = new Matrix([[4, 1, 2], [6, 3, 0]]); 