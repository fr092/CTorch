#include <iostream>
#include <armadillo>
#include "layers.hpp"
int main(){

    Tensor* x = new Tensor();
    x->create(4, 4);
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++)
            x->data(i, j) = 0.123*i;
    }

    Tensor* y = new Tensor();
    y->create(4, 1);
    for(int i=0;i<4;i++)
        y->data(i, 0) = 1 + 0.1*i;

    x->isgrad = false;
    y->isgrad = false;

    Tensor* ypred = new Tensor();

    Linear *input = new Linear(4, 50);
    Linear *output = new Linear(50, 1);

    std::vector<Tensor*> params;
    input->parameters(params);
    output->parameters(params);

    int epochs = 100;

    Tensor* loss = new Tensor();
    Tensor* x_temp = new Tensor();
    params.push_back(x_temp);

    for(int i=0;i<epochs;i++){
        
        x_temp->operator=(x);
        x_temp = input->call(x_temp);
        x_temp = output->call(x_temp);

        loss = x_temp->mse(y);
        
        for(auto i:params)
            i->set_grad_to_zero();
        
        loss->backward();
        loss->backpropagate(0.01, params);
    }

    ypred->operator=(x_temp);

    for(int i=0;i<ypred->data.n_rows;i++){
        for(int j=0;j<ypred->data.n_cols;j++)
            std::cout<<ypred->data(i, j)<<" ";
        std::cout<<std::endl;
    }

    return 0;
}