#include <iostream>
#include <armadillo>
#include <vector>
#include <stack>
#include <map>

class Tensor{

    private:

    public:

    bool isgrad = true;
    arma::dmat data;
    arma::dmat grad;
    std::vector<Tensor*> prev;
    void(*_backward)(std::vector<Tensor*>,Tensor*) = nullptr;

    void create(int rows, int cols, std::string ini="false", double scalar=0, std::string grad="true"){
        
        if(grad == "true")
            this->grad.zeros(rows, cols);

        if(grad == "false")
            this->isgrad = false;

        if(ini == "norm")
            this->data.randn(rows, cols);

        else if(ini == "unif")
            this->data.randn(rows, cols);

        else if(ini == "false")
            this->data = arma::dmat(rows, cols, arma::fill::value(scalar));

        return ;

    }

    void set_data(double scalar){
        this->data.fill(scalar); 
        return ;
    }

    Tensor* operator+(Tensor* other){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        
        if(this->data.n_rows != other->data.n_rows && other->data.n_rows == 1){

            result->data = this->data.each_row() + other->data;
            
            if(isgrad == true){

                auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                    Tensor* first = *prev.begin();
                    Tensor* second = *(prev.end()-1);
                    first->grad += result->grad;
                    second->grad += arma::sum(result->grad, 0);
                    return ;
                };

                result->_backward = _backward;

            }
        }
        
        else if(this->data.n_cols != other->data.n_cols && other->data.n_cols == 1){

            result->data = this->data.each_col() + other->data;

            if(isgrad == true){

                auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                    Tensor* first = *prev.begin();
                    Tensor* second = *(prev.end()-1);
                    first->grad += result->grad;
                    second->grad += arma::sum(result->grad, 1);
                    return ;
                };

                result->_backward = _backward;

            }
        }
        
        else{

            result->data = this->data + other->data;
            
            if(isgrad == true){

                auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                    Tensor* first = *prev.begin();
                    Tensor* second = *(prev.end()-1);
                    first->grad += result->grad;
                    second->grad += result->grad;
                    return ;
                };
                
                result->_backward = _backward;

            }
        }

        if(isgrad == true){
            result->prev.push_back(this);
            result->prev.push_back(other);
        }
        
        return result;
    }

    Tensor* operator-(Tensor* other){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        
        if(this->data.n_rows != other->data.n_rows && other->data.n_rows == 1){
            
            result->data = this->data.each_row() - other->data;

            if(isgrad == true){
                
                auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                    Tensor* first = *prev.begin();
                    Tensor* second = *(prev.end()-1);
                    first->grad += result->grad;
                    second->grad -= arma::sum(result->grad, 0);
                    return ;
                };

                result->_backward = _backward;

            }
        }
        
        else if(this->data.n_cols != other->data.n_cols && other->data.n_cols == 1){
            
            result->data = this->data.each_col() - other->data;
            
            if(isgrad == true){

                auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                    Tensor* first = *prev.begin();
                    Tensor* second = *(prev.end()-1);
                    first->grad += result->grad;
                    second->grad -= arma::sum(result->grad, 1);
                    return ;
                };

                result->_backward = _backward;

            }
        }
        
        else{
            
            result->data = this->data - other->data;

            if(isgrad == true){

                auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                    Tensor* first = *prev.begin();
                    Tensor* second = *(prev.end()-1);
                    first->grad += result->grad;
                    second->grad -= result->grad;
                    return ;
                };
                
                result->_backward = _backward;

            }
        }
        
        if(isgrad == true){
            result->prev.push_back(this);
            result->prev.push_back(other);
        }
        
        return result;
    }

    Tensor* operator*(Tensor* other){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        result->data = this->data % other->data;

        if(isgrad == true){

            result->prev.push_back(this);
            result->prev.push_back(other);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                Tensor* second = *(prev.end()-1);
                first->grad += (second->data % result->grad);
                second->grad += (first->data % result->grad);
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* operator/(Tensor* other){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        result->data = this->data / other->data;

        if(isgrad == true){
            
            result->prev.push_back(this);
            result->prev.push_back(other);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                Tensor* second = *(prev.end()-1);
                first->grad += (result->grad / second->data);
                second->grad -= ((first->data % result->grad) / arma::pow(second->data, 2));
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* log(){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        result->data = arma::log(this->data);

        if(isgrad == true){

            result->prev.push_back(this);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                first->grad += (result->grad / first->data);
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* exp(){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        result->data = arma::exp(this->data);

        if(isgrad == true){

            result->prev.push_back(this);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                first->grad += (result->data % result->grad);
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* pow(double scalar){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        result->data = arma::pow(this->data, scalar);
        Tensor* temp = new Tensor;
        temp->data = scalar;

        if(isgrad == true){
            
            result->prev.push_back(this);
            result->prev.push_back(temp);

            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                Tensor* second = *(prev.end()-1);
                first->grad += (second->data(0, 0) * ((result->data / first->data) % result->grad));
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* mat_mul(Tensor* other){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        result->data = this->data * other->data;

        if(isgrad = true){
            
            result->prev.push_back(this);
            result->prev.push_back(other);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                Tensor* second = *(prev.end() - 1);
                arma::dmat temp1 = second->data.t();
                arma::dmat temp2 = first->data.t();
                first->grad += result->grad * temp1;
                second->grad += temp2 * result->grad;
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* operator^(int scalar){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        result->data = powmat(this->data, scalar);
        if(isgrad){

            result->prev.push_back(this);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* tanh(){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        result->data = arma::tanh(this->data);
        if(isgrad == true){

            result->prev.push_back(this);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                first->grad += ((1 - arma::pow(result->data, 2)) % result->grad);
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* ReLU(){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        double max = this->data.max();
        result->data = arma::clamp(this->data, 0, max);
        if(isgrad == true){

            result->prev.push_back(this);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                for(int i=0;i<first->data.n_rows;i++){
                    for(int j=0;j<first->data.n_cols;j++){
                        if(result->data(i,j)>0)
                            first->grad(i,j) += result->grad(i,j);
                    }
                }
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* sigmoid(){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        arma::dmat temp = this->data * (-1);
        temp = arma::exp(temp);
        temp = temp + 1;
        temp = 1/temp;
        result->data = temp;

        if(isgrad == true){

            result->prev.push_back(this);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                first->grad += ((result->data % (1 - result->data)) % result->grad);
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* softmax(){
        
        Tensor* result = new Tensor;
        result->create(this->data.n_rows, this->data.n_cols);
        arma::dmat temp = arma::exp(this->data) ;
        arma::dmat sum = arma::sum(temp, 1);
        result->data = temp.each_col()/sum;
        if(isgrad == true){

            result->prev.push_back(this);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *prev.begin();
                arma::dmat temp = first->grad;
                for(int i=0;i<temp.n_rows;i++){
                    for(int j=0;j<temp.n_cols;j++){
                        for(int k=0;k<temp.n_cols;k++){
                            if(j==k)
                                temp(i,j) -= ((result->data(i,k))*(1-result->data(i,k)));
                            else
                                temp(i,j) -= (result->data(i,j)*result->data(i,k));
                        }
                    }
                }
                first->grad = temp % result->grad;
                return ;
            };
            
            result->_backward = _backward;

        }
        
        return result;
    }

    Tensor* mse(Tensor* y){

        Tensor* result = new Tensor;
        result->create(1, 1);

        for(int i=0;i<y->data.n_rows;i++){
            int x = (this->data(i, 0) - y->data(i, 0));
            result->data(0, 0) += (x * x);
        }

        result->data(0, 0) /= y->data.n_rows; 

        if(isgrad == true){

            result->prev.push_back(this);
            
            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *(prev.begin());
            };

            result->_backward = _backward;

        }

        return result;
    }

    Tensor* cross_entropy(Tensor* y){

        Tensor* result = new Tensor;
        result->create(1, 1);

        arma::dmat temp = this->data;
        temp = arma::exp(temp);
        temp = temp.each_col() / arma::sum(temp, 1);

        arma::dmat result_probs;
        result_probs.zeros(y->data.n_rows, 1);

        for(int i=0;i<y->data.n_rows;i++)
            result_probs(i, 0) = temp(i, y->data(i, 0));

        result->data = arma::sum(result_probs, 0);
        result->data /= y->data.n_rows;

        if(isgrad == true){

            result->prev.push_back(this);
            result->prev.push_back(y);

            auto _backward = [](std::vector<Tensor*> prev, Tensor* result){
                Tensor* first = *(prev.begin());
                Tensor* second = *(prev.end()-1);
                arma::dmat temp = arma::exp(first->data);
                temp = temp.each_col() / arma::sum(temp, 1);
                for(int i=0;i<first->data.n_rows;i++)
                    temp(i, second->data(i, 0)) -= 1;
                temp /= second->data.n_rows;
                first->grad += temp;
            };

            result->_backward = _backward;

        }

        return result;
    }

    void build_topo(Tensor* node, std::stack<Tensor*> &topo, std::map<Tensor*,bool> &visited){

        if(visited[node] == false){
            
            visited[node] = true;

            for(auto i:node->prev){
                if(i->isgrad == true)
                    build_topo(i, topo, visited);
            }
            
            topo.push(node);

        }

        return ;
    }

    void backward(){

        std::stack<Tensor*> topo;
        std::map<Tensor*,bool> visited;

        build_topo(this, topo, visited);

        this->grad += 1;

        while(topo.empty() == false){
            Tensor* curr = topo.top();
            topo.top()->_backward(curr->prev, curr);
        }

        return ;
    }

};