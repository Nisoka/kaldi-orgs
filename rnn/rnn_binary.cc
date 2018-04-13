#include "rnn_binary.h"

using namespace std;


int largest_number = (pow(2, binary_dim));

double sigmoid(double x){
    return (1.0/(1.0 + exp(-x)));
}

double dsigmoid(double y){
    return y*(1-y);
}


void int2binary(int n, int *arr){
    int i = 0;
    while(n){
        arr[i++] = n%2;
        n /= 2;
    }

    while(i < binary_dim){
        arr[i++] = 0;
    }
}

void binary2int(int *arr, int dim, int &n){

    dim -= 1;
    while(dim >= 0){
        n += arr[dim] * pow(2, dim);
        dim --;
    }
}



void W_init(double w[], int n){
    for(int i = 0; i < n; i++){
        w[i] = uniform_plus_minus_one;
    }
}


RNN::RNN(){
    layer_0 = new double[innode];
    layer_2 = new double[outnode];

    W_init((double*)U, innode*hiddenode);
    W_init((double*)V, hiddenode*outnode);
    W_init((double*)W, hiddenode*hiddenode);
}

RNN::~RNN(){
    delete layer_0;
    delete layer_2;
}


void RNN::train(){

    int epoch, i, j, k, m, p;
    vector<double*> layer_1_vector;
    vector<double> layer_2_delte;

    for(epoch = 0; epoch < 100000; epoch++){
        double e = 0.0;
        int len = layer_1_vector.size();
        for(i = 0; i < len; i++)
            delete layer_1_vector[i];
        layer_1_vector.clear();
        layer_2_delte.clear();

        int d[binary_dim];
        memset(d, 0, sizeof(d));

        int a_int = (int)randval(largest_number/2.0);
        int a[binary_dim];
        int2binary(a_int, a);

        int b_int = (int)randval(largest_number/2.0);
        int b[binary_dim];
        int2binary(b_int, b);

        int c_int = a_int + b_int;
        int c[binary_dim];
        int2binary(c_int, c);

        // HIDE_ACTIVE(t = -1) 是无信息, 为满足rnn的隐藏层Time方向链接结构
        layer_1 = new double[hiddenode];
        for(i = 0; i < hiddenode; i++){
            layer_1[i] = 0;
        }
        layer_1_vector.push_back(layer_1);



        for(p = 0; p< binary_dim; p++) {
            layer_0[0] = a[p];
            layer_0[1] = b[p];
            double y = (double)c[p];



            // HIDE = W* HIDE_ACTIVE(t-1) + U* input
            // HIDE_ACTIVE = SIGMOID(HIDE)

            // HIDE_ACTIVE --
            layer_1 = new double[hiddenode];

            for(j = 0; j < hiddenode; j++){
                // HIDE_J
                double hj = 0.0;
                // HIDE_J  += U * input
                for(m = 0; m < innode; m ++){
                    hj += layer_0[m] * U[m][j];
                }
                // HIDE_J += W * HIDE_ACTIVE(t-1)
                double *layer_1_pre = layer_1_vector.back();
                for(m = 0; m < hiddenode; m++){
                    hj += layer_1_pre[m] * W[m][j];
                }

                // HIDE_ACTIVE_J = SIGMOID(HIDE_J)
                layer_1[j] = sigmoid(hj);
            }

            // HIDE_ACTIVE  (t = 0, 1,... 7)
            layer_1_vector.push_back(layer_1);



            // OUT = V* HIDE
            // y = SIGMOID(HIDE)
            for(k = 0; k < outnode; k++){
                // OUT_k
                double out = 0.0;
                // OUT_k = V * HIDE_ACTIVE
                for(m = 0; m < hiddenode; m ++){
                    out += layer_1[m] * V[m][k];
                }

                // y_k = SIGMOID(OUT_k)
                layer_2[k] = sigmoid(out);
            }



            // 残差 计算, 不能出错,不能使用 d[p] 代替 layer_2[0]
            // 不然会丢失残差. 因为残差是要用来 计算各个参数贡献的值, 如果进行修改, 那么会导致
            // 贡献度 -- 即梯度出错, 必然导致参数更新出错.
            // y 是0/1 实际输出向量只有len=1, 记录输出值
            d[p] = (int)floor(layer_2[0] + 0.5);
//            cout << d[p] << " : "<< layer_2[0] << " : " << c[p] << endl;
            // 误差 y^ - y
            double err = y-layer_2[0];
            e += fabs(err);

            // OUT 残差 == (y^-y)*(deriv_y_for_out)
            layer_2_delte.push_back(err*dsigmoid(layer_2[0]));

        }





        //残差反向传播




        // HIDE_ACTIVE 残差
        double *layer_1_delte = new double[hiddenode];
        // HIDE_ACTIVE(t = TIME+1) 残差, TIME最后序列没有后继的贡献, 满足结构隐层贡献为0
        double *layer_1_futher_delte = new double[hiddenode];
        for(j = 0; j < hiddenode; j++)
            layer_1_futher_delte[j] = 0;

        //time 逐个sample.
        for(p = binary_dim-1; p >= 0; p--){
            layer_0[0] = a[p];
            layer_0[1] = b[p];

            // HIDE_ACTIVE(t --- now)
            // 因为layer_1_vector 首个元素 保存的是为t=0的之前隐层信息(实际没有信息)
            layer_1 = layer_1_vector[p+1];
            // HIDE_ACTIVE(t-1)
            double *layer_1_pre = layer_1_vector[p];


            // Y = SIGMOID(OUT)
            // OUT = V* HIDE_ACTIVE
            // OUT残差 = layer_2_delte[p]
            //    通过OUT残差 由 V 贡献为 derive_OUT_for_V
            for(k = 0; k < outnode; k++){
                for(j = 0; j < hiddenode; j++){
                    V[j][k] += alpha * layer_2_delte[p] * layer_1[j];
                }
            }


            // OUT = V* HIDE_ACTIVE
            // HIDE(t+1) = W* HIDE_ACTIVE + U* x

            // HIDE = SIGMOID(HIDE_ACTIVE)

            // HIDE_ACTIVE 的残差为
            // HIDE_ACTIVE_DELTE = derive_Out_for_HIDE_ACTIVE + derive_HIDE(t+1)_for_HIDE_ACTIVE
            for(j = 0; j < hiddenode; j++){
                layer_1_delte[j] = 0.0;
                // 隐藏层HIDE_ACTIVE残差(from 结构反向传播 derive_OUT_for_HIDE_ACTIVE)
                for(k = 0; k < outnode; k++){
                    layer_1_delte[j] += layer_2_delte[p] * V[j][k];
                }
                // 隐藏层HIDE_ACTIVE残差(from Time反向传播 derive_HIDE(t+1)_for_HIDE_ACTIVE)
                for(k = 0; k < hiddenode; k++){
                    layer_1_delte[j] += layer_1_futher_delte[k]*W[j][k];
                }

                // HIDE = SIGMOID(HIDE_ACTIVE)
                // 则 HIDE 隐藏层残差为 --
                layer_1_delte[j] = layer_1_delte[j]* dsigmoid(layer_1[j]);


                // HIDE = U*x + W* HIDE_ACTIVE(t-1)
                // 更新U W , 分别是从HIDE的残差求解得到的 U W的贡献.
                for(k = 0; k < innode; k++){
                    U[k][j] += alpha * layer_1_delte[j] * layer_0[k];
                }
                for(k = 0; k < hiddenode; k++){
                    W[k][j] += alpha * layer_1_delte[j] * layer_1_pre[k];
                }
            }

            // 特殊的残差只对最终节点有效, 用后即焚
            if(p == binary_dim - 1)
                delete layer_1_futher_delte;
            // 保留t+1 的 HIDE 残差, 用于计算t的 HIDE_ACTIVE残差.
            layer_1_futher_delte = layer_1_delte;
        }

        delete layer_1_futher_delte;


        if(epoch % 200 == 0){
            cout << "error : " << e << endl;
            cout << "pred : ";
            for(k = binary_dim-1; k >= 0; k--)
                cout << d[k];
            cout << endl;

            cout << "true : ";
            for(k = binary_dim-1; k >= 0; k--)
                cout << c[k];
            cout << endl;

            int out = 0;
            binary2int(d, binary_dim, out);

            cout << a_int << " + " << b_int << ": true is " << c_int << " pred is " << out << endl;

        }
    }
}




