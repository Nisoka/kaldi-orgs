//让程序自己学会是否需要进位，从而学会加法
#include "cstring"
#include "iostream"
#include "math.h"
#include "stdlib.h"
#include "time.h"
#include "vector"
#include "assert.h"
using namespace std;

#define innode  2       //输入结点数，将输入2个加数
#define hidenode  16    //隐藏结点数，存储“携带位”
#define outnode  1      //输出结点数，将输出一个预测数字
#define alpha  0.1      //学习速率
#define binary_dim 8    //二进制数的最大长度

#define randval(high) ( (double)rand() / RAND_MAX * high )
#define uniform_plus_minus_one ( (double)( 2.0 * rand() ) / ((double)RAND_MAX + 1.0) - 1.0 )  //均匀随机分布


int largest_number = ( pow(2, binary_dim) );  //跟二进制最大长度对应的可以表示的最大十进制数

//激活函数
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

//激活函数的导数，y为激活函数值
double dsigmoid(double y)
{
    return y * (1 - y);
}
//将一个10进制整数转换为2进制数
void int2binary(int n, int *arr)
{
    int i = 0;
    while(n)
    {
        arr[i++] = n % 2;
        n /= 2;
    }
    while(i < binary_dim)
        arr[i++] = 0;
}

class RNN
{
public:
    RNN();
    virtual ~RNN();
    void train();

public:
    double w[innode][hidenode];        //连接输入层与隐藏层的权值矩阵
    double w1[hidenode][outnode];      //连接隐藏层与输出层的权值矩阵
    double wh[hidenode][hidenode];     //连接前一时刻的隐含层与现在时刻的隐含层的权值矩阵

    double *layer_0;       //layer 0 输出值，由输入向量直接设定
    //double *layer_1;     //layer 1 输出值
    double *layer_2;       //layer 2 输出值
};

void winit(double w[], int n) //权值初始化
{
    for(int i=0; i<n; i++)
        w[i] = uniform_plus_minus_one;  //均匀随机分布
}

RNN::RNN()
{
    layer_0 = new double[innode];
    layer_2 = new double[outnode];
    winit((double*)w, innode * hidenode);
    winit((double*)w1, hidenode * outnode);
    winit((double*)wh, hidenode * hidenode);
}

RNN::~RNN()
{
    delete layer_0;
    delete layer_2;
}

void RNN::train()
{
    int epoch, i, j, k, m, p;
    vector<double*> layer_1_vector;      //保存隐藏层
    vector<double> layer_2_delta;        //保存误差关于Layer 2 输出值的偏导

    for(epoch=0; epoch<100000; epoch++)  //训练次数
    {
        double e = 0.0;  //误差
        for(i=0; i<layer_1_vector.size(); i++)
            delete layer_1_vector[i];
        layer_1_vector.clear();
        layer_2_delta.clear();

        int d[binary_dim];                    //保存每次生成的预测值
        memset(d, 0, sizeof(d));

        int a_int = (int)randval(largest_number/2.0);  //随机生成一个加数 a
        int a[binary_dim];
        int2binary(a_int, a);                 //转为二进制数

        int b_int = (int)randval(largest_number/2.0);  //随机生成另一个加数 b
        int b[binary_dim];
        int2binary(b_int, b);                 //转为二进制数

        int c_int = a_int + b_int;            //真实的和 c
        int c[binary_dim];
        int2binary(c_int, c);                 //转为二进制数

        double *layer_1 = new double[hidenode];
        for(i=0; i<hidenode; i++)         //在0时刻是没有之前的隐含层的，所以初始化一个全为0的
            layer_1[i] = 0;
        layer_1_vector.push_back(layer_1);

        //正向传播
        for(p=0; p<binary_dim; p++)           //循环遍历二进制数组，从最低位开始
        {
            layer_0[0] = a[p];
            layer_0[1] = b[p];
            double y = (double)c[p];          //实际值
            layer_1 = new double[hidenode];   //当前隐含层

            for(j=0; j<hidenode; j++)
            {
                //输入层传播到隐含层
                double o1 = 0.0;
                for(m=0; m<innode; m++)
                    o1 += layer_0[m] * w[m][j];

                //之前的隐含层传播到现在的隐含层
                double *layer_1_pre = layer_1_vector.back();
                for(m=0; m<hidenode; m++)
                    o1 += layer_1_pre[m] * wh[m][j];

                layer_1[j] = sigmoid(o1);      //隐藏层各单元输出
            }

            for(k=0; k<outnode; k++)
            {
                //隐藏层传播到输出层
                double o2 = 0.0;
                for(j=0; j<hidenode; j++)
                    o2 += layer_1[j] * w1[j][k];
                layer_2[k] = sigmoid(o2);          //输出层各单元输出
            }

            d[p] = (int)floor(layer_2[0] + 0.5);   //记录预测值
            layer_1_vector.push_back(layer_1);     //保存隐藏层，以便下次计算

            //保存标准误差关于输出层的偏导
            layer_2_delta.push_back( (y - layer_2[0]) * dsigmoid(layer_2[0]) );
            e += fabs(y - layer_2[0]);          //误差
        }

        //误差反向传播

        //隐含层偏差，通过当前之后一个时间点的隐含层误差和当前输出层的误差计算
        double *layer_1_delta = new double[hidenode];
        double *layer_1_future_delta = new double[hidenode];   //当前时间之后的一个隐藏层误差
        for(j=0; j<hidenode; j++)
            layer_1_future_delta[j] = 0;
        for(p=binary_dim-1; p>=0 ; p--)
        {
            layer_0[0] = a[p];
            layer_0[1] = b[p];

            layer_1 = layer_1_vector[p+1];     //当前隐藏层
            double *layer_1_pre = layer_1_vector[p];   //前一个隐藏层

            for(k=0; k<outnode; k++)  //对于网络中每个输出单元，更新权值
            {
                //更新隐含层和输出层之间的连接权
                for(j=0; j<hidenode; j++)
                    w1[j][k] += alpha * layer_2_delta[p] * layer_1[j];
            }

            for(j=0; j<hidenode; j++) //对于网络中每个隐藏单元，计算误差项，并更新权值
            {
                layer_1_delta[j] = 0.0;
                for(k=0; k<outnode; k++)
                    layer_1_delta[j] += layer_2_delta[p] * w1[j][k];
                for(k=0; k<hidenode; k++)
                    layer_1_delta[j] += layer_1_future_delta[k] * wh[j][k];

                //隐含层的校正误差
                layer_1_delta[j] = layer_1_delta[j] * dsigmoid(layer_1[j]);

                //更新输入层和隐含层之间的连接权
                for(k=0; k<innode; k++)
                    w[k][j] += alpha * layer_1_delta[j] * layer_0[k];

                //更新前一个隐含层和现在隐含层之间的权值
                for(k=0; k<hidenode; k++)
                    wh[k][j] += alpha * layer_1_delta[j] * layer_1_pre[k];
            }

            if(p == binary_dim - 1)
                delete layer_1_future_delta;
            layer_1_future_delta = layer_1_delta;
        }
        delete layer_1_future_delta;

        if(epoch % 1000 == 0)
        {
            cout << "error：" << e << endl;
            cout << "pred：" ;
            for(k=binary_dim-1; k>=0; k--)
                cout << d[k];
            cout << endl;

            cout << "true：" ;
            for(k=binary_dim-1; k>=0; k--)
                cout << c[k];
            cout << endl;

            int out = 0;
            for(k=binary_dim-1; k>=0; k--)
                out += d[k] * pow(2, k);
            cout << a_int << " + " << b_int << " = " << out << endl << endl;
        }
    }
}


int main()
{
    srand(time(NULL));
    RNN rnn;
    rnn.train();
    return 0;
}
