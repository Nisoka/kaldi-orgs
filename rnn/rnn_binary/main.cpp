#include <iostream>
#include <time.h>
#include "rnn_binary.h"

using namespace std;

int main()
{

    srand(time(NULL));
    RNN rnn;
    rnn.train();

    cout << "Hello World!" << endl;
    return 0;
}
