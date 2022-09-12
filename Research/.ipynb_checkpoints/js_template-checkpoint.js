/* dActs */
#{act_fncs}#

class Model{
    constructor(){
        // TODO : Convert each array to matrix class
        this.arrW = #{l_w}#;
        this.arrB = #{l_b}#;
        this.arrA = #{l_a}#;
        this.numLayers = this.arrW.length;
    }
        
    pred(x){
        let output = new Matrix([...x]);
        
        for (let idx = 0; idx < this.numLayers; idx++){
            const actFuntion = dActs[this.arrA[idx]];
            const weights = this.arrW[idx];
            const bias = this.arrB[idx];
            if ( bias != 0 ){
                output = actFuntion(output.dot(weights).add(bias));
            }else{
                output = actFuntion(output.dot(weights));
            }
        }
        return output;
    }
}