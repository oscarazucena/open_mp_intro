#include <omp.h>
#include <iostream>
#include <map>

#include <functional>

using namespace std;

template<int N>
class SimpleIntegrator
{
protected:
    const std::function<double(double)> &f;
    size_t size;
    double start;
    double end;
    int part;
    const double getRange()
    {
        return end-start;
    }
    const double getPart()
    {
        return part;
    }
public:
    SimpleIntegrator(const std::function<int(int)> &f_in, const size_t &size_in, const double &start_in, const double &end_in) : f(f_in),
        size(size_in),
        start(start_in),
        end(end_in),
        part(size/N)
    {
    }
    virtual ~SimpleIntegrator()
    {

    }
    virtual double run() = 0;
};

template<int N>
class SimpleIntegratorArray : public SimpleIntegrator<N>
{
    double sum_array[N];
public:
    SimpleIntegratorArray(const std::function<int(int)> &f_in, const size_t &size_in, const double &start_in, const double &end_in) : SimpleIntegrator<N>(f_in,size_in, start_in,end_in)
    {

    }
    double run() override
    {
        omp_set_num_threads(N);
        double temp_del = this->getRange()/this->size;
#pragma omp parallel
        {
            double sum_temp = 0.0;
            int id = omp_get_thread_num();
            int temp_N = this->getPart()*(id+1);
            double x_temp =  this->getPart()*id*temp_del + temp_del/2;
            for(int c =  this->getPart()*id; c < temp_N; c++)
            {
                sum_temp += 4.0/(1.0 + x_temp*x_temp);
                x_temp += temp_del;
            }
            sum_array[id] = sum_temp;
        }
        double sum = 0;
        for(const auto & val : sum_array)
        {
            sum+=val;
        }
        sum *= temp_del;
        return sum;
    }
};

template<int N>
class SimpleIntegratorAtomic : public SimpleIntegrator<N>
{
    double sum;
public:
    SimpleIntegratorAtomic(const std::function<int(int)> &f_in, const size_t &size_in, const double &start_in, const double &end_in) : SimpleIntegrator<N>(f_in,size_in, start_in,end_in),
        sum(0.0f)
    {

    }
    double run() override
    {
        omp_set_num_threads(N);
        double temp_del = this->getRange()/this->size;
#pragma omp parallel
        {
            double sum_temp = 0.0;
            int id = omp_get_thread_num();
            int temp_N = this->getPart()*(id+1);
            double x_temp =  this->getPart()*id*temp_del + temp_del/2;
            for(int c =  this->getPart()*id; c < temp_N; c++)
            {
                sum_temp += 4.0/(1.0 + x_temp*x_temp);
                x_temp += temp_del;
            }
#pragma omp atomic update
            sum += sum_temp;
        }

        sum *= temp_del;
        return sum;
    }
};

int main()
{
    map<int,double> time_map
    {
        {1,0.0f},
        {2,0.0f},
        {4,0.0f},
        {8,0.0f},
    };

    int loop_total = 100;
    //Equation for pi is : sum(0,1,deltaX,F(x)*deltaX)
    //where F(x) = 4.0/(1+x*x)
    const auto F = [](const double x)->double
    {
        return 4.0/(1.0 + x*x);
    };

    const size_t N = 10000000; // use 100 sub divs
    const double begin = 0.0f;
    const double end = 1.0f;

    for(int count = 0; count < loop_total; count++)
    {


        double start = omp_get_wtime();
        SimpleIntegratorAtomic<1> integrator{F,N,begin,end};
        double sum = integrator.run();
        time_map[1] += omp_get_wtime() - start;
        cout << "----------------------------------------" << endl;
        cout << "Pi: " << sum << endl;

        start = omp_get_wtime();
        SimpleIntegratorAtomic<2> integrator2{F,N,begin,end};
        sum = integrator2.run();
        time_map[2] += omp_get_wtime() - start;
        cout << "Pi: " << sum << endl;

        start = omp_get_wtime();
        SimpleIntegratorAtomic<4> integrator4{F,N,begin,end};
        sum = integrator4.run();
        time_map[4] += omp_get_wtime() - start;
        cout << "Pi: " << sum << endl;

        start = omp_get_wtime();
        SimpleIntegratorAtomic<8> integrator8{F,N,begin,end};
        sum = integrator8.run();
        time_map[8] += omp_get_wtime() - start;
        cout << "Pi: " << sum << endl;
    }

    for (auto pair : time_map)
    {
        cout << "Number of threads: " << pair.first << endl;
        cout << "Mean time: " << pair.second/(loop_total)*1000 << " ms" << endl;
    }

    return 0;
}
