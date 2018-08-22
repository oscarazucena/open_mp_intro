#include <omp.h>
#include <iostream>

using namespace std;
int main()
{
#pragma omp parallel
    {
        //could not use cout as it is not thread safe
        int id = omp_get_thread_num();
        printf("Hello From Thread: %d\n",id);
        printf("World (%d)\n", id );
    }

    return 0;
}
