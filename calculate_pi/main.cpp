#include <omp.h>
#include <iostream>
#include <map>

using namespace std;
int main()
{
    map<int,double> time_map
    {
        {0,0.0f},
        {2,0.0f},
        {4,0.0f},
        {8,0.0f},
    };
    int loop_total = 100;
    for(int count = 0; count < loop_total; count++)
    {
        //Equation for pi is : sum(0,1,deltaX,F(x)*deltaX)
        //where F(x) = 4.0/(1+x*x)
        double start = omp_get_wtime();
        int N = 10000000; // use 100 sub divs
        double range = 1.0;
        double deltaX = range/double(N);
        double sum = 0;
        auto F = [](double x)->double
        {
            return 4.0/(1.0 + x*x);
        };
        double x = deltaX/2;
        for(int c = 0; c < N; c++)
        {
            sum += 4.0/(1.0 + x*x)*deltaX;
            x += deltaX;
        }

        time_map[0] += omp_get_wtime() - start;

        omp_set_num_threads(2);
        int part = N/2;
        sum = 0;
        start = omp_get_wtime();
#pragma omp parallel
        {
            double sum_temp = 0.0;
            int id = omp_get_thread_num();
            double temp_del = deltaX;
            int temp_N = part*(id+1);
            double x_temp = part*id*temp_del + temp_del/2;
            for(int c = part*id; c < temp_N; c++)
            {
                sum_temp += 4.0/(1.0 + x_temp*x_temp)*temp_del;
                x_temp += temp_del;
            }
            sum += sum_temp;
            //printf("Id: %d\n", id);
        }
        time_map[2] += omp_get_wtime() - start;

        omp_set_num_threads(4);
        part = N/4;
        sum = 0;
        start = omp_get_wtime();
#pragma omp parallel
        {
            double sum_temp = 0.0;
            int id = omp_get_thread_num();
            double temp_del = deltaX;
            int temp_N = part*(id+1);
            double x_temp = part*id*temp_del + temp_del/2;
            for(int c = part*id; c < temp_N; c++)
            {
                sum_temp += 4.0/(1.0 + x_temp*x_temp)*temp_del;
                x_temp += temp_del;
            }
            sum += sum_temp;
            //printf("Id: %d\n", id);
        }
        time_map[4] += omp_get_wtime() - start;

        omp_set_num_threads(8);
        part = N/8;
        sum = 0;
        start = omp_get_wtime();
#pragma omp parallel
        {
            double sum_temp = 0.0;
            int id = omp_get_thread_num();
            double temp_del = deltaX;
            int temp_N =  part*(id+1);
            double x_temp = part*id*temp_del + temp_del/2;
            for(int c = part*id; c < temp_N; c++)
            {
                sum_temp += 4.0/(1.0 + x_temp*x_temp)*temp_del;
                x_temp += temp_del;
            }
            sum += sum_temp;
            //printf("Id: %d\n", id);
        }
        time_map[8] += omp_get_wtime() - start;
    }

    for (auto pair : time_map)
    {
        cout << "Number of threads: " << pair.first << endl;
        cout << "Mean time: " << pair.second/(loop_total)*1000 << " ms" << endl;
    }

    return 0;
}
