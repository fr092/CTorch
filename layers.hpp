#include<iostream>
#include<vector>
#include "tensor.hpp"

class Linear{
    
    private:

    public:

    Tensor* weight = new Tensor;
    Tensor* bias = new Tensor;
    bool is_bias = true;

    Linear(int fan_in, int fan_out, bool if_bias=true){
        
        this->weight->create(fan_in, fan_out, "norm");
        this->weight->data /= sqrt(fan_in);
        if(if_bias)
            this->bias->create(1, fan_out, "false", 0);
        else{
            delete(bias);
            is_bias = false;    
        }
    }
    
    Tensor* call(Tensor* x){

        Tensor* out = new Tensor;
        
        if(is_bias)
            out = (x->mat_mul(weight))->operator+(bias);
        else 
            out = x->mat_mul(weight);
        
        return out;
    }

    std::vector<Tensor*> parameters(){
        
        std::vector<Tensor*> out;
        out.push_back(weight);
        if(is_bias)
            out.push_back(bias);
        
        return out;
    }

};

class RNN{

    private:

    public:

    Tensor* input_weight = new Tensor;
    Tensor* hidden_weight = new Tensor;
    Tensor* hidden_bias = new Tensor;
    Tensor* output_weight = new Tensor;
    Tensor* output_bias = new Tensor;

    RNN(int input_features, int hidden_features, int output_features){

        (this->input_weight)->create(input_features, hidden_features, "norm");
        (this->hidden_weight)->create(hidden_features, hidden_features, "norm");
        (this->hidden_bias)->create(1, hidden_features);
        (this->output_weight)->create(hidden_features, output_features, "norm");
        (this->output_bias)->create(1, output_features);

    }

    Tensor* call(Tensor* x){
        
        Tensor* out = new Tensor;
        Tensor* hidden = new Tensor;
        Tensor* temp = new Tensor;
        Tensor* xi = new Tensor;
        hidden->create(1, hidden_weight->data.n_cols);
        temp->create(1, x->data.n_cols);
        xi->create(1, hidden_weight->data.n_cols);
        
        for(int i=0;i<x->data.n_rows;i++){      
            
            for(int j=0;j<x->data.n_cols;j++)
                temp->data(0, j) = x->data(i, j);
            
            xi = temp->mat_mul(input_weight);
            hidden = hidden->mat_mul(hidden_weight);
            hidden = hidden->operator+(hidden_bias);
            hidden = xi->operator+(hidden);
            hidden = hidden->tanh();

        }

        out = out->mat_mul(output_weight);
        out = out->operator+(output_bias);
        
        return out;
    }

    std::vector<Tensor*> parameters(){

        std::vector<Tensor*> out;
        out.push_back(input_weight);
        out.push_back(hidden_weight);
        out.push_back(hidden_bias);
        out.push_back(output_weight);
        out.push_back(output_bias);
        
        return out;
    }

};

class LSTM{
    
};