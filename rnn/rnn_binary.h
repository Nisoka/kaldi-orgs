#ifndef RUN_BINARY_H
#define RUN_BINARY_H


#define innode 2
#define hiddenode 16
#define outnode 1
#define alpha 0.1
#define binary_dim 8

#define randval(high) ( ((double)rand() / RAND_MAX ) * high )
#define uniform_plus_minus_one ( (double)(2.0 * rand()) / ((double)RAND_MAX + 1.0) - 1.0 )



class RNN
{
public:
    RNN();
    virtual ~RNN();
    void train();


public:
    // HIDE = V* HIDE_ACTIVE(t-1) + U* x
    // HIDE_ACTIVE = SIGMOID(HIDE_IN)
    // OUT = W* HIDE_ACTIVE
    // Y = SIGMOID(OUT)
    double U[innode][hiddenode];
    double V[hiddenode][outnode];
    double W[hiddenode][hiddenode];

    double *layer_0;
    double *layer_1;
    double *layer_2;
};
#endif // RUN_BINARY_H
