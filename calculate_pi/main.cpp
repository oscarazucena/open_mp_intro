#include <omp.h>
#include <iostream>
#include <map>

#include <functional>

using namespace std;

//templated class to call integrator with different number of processors
//N is the number of processors
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
    SimpleIntegrator(const std::function<double(double)> &f_in, const size_t &size_in, const double &start_in, const double &end_in) :
        f(f_in),
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

//Integrator that uses array to store the results
template<int N>
class SimpleIntegratorArray : public SimpleIntegrator<N>
{
    double sum_array[N];
public:
    SimpleIntegratorArray(const std::function<double(double)> &f_in, const size_t &size_in, const double &start_in, const double &end_in) : SimpleIntegrator<N>(f_in,size_in, start_in,end_in)
    {
        cout << "SimpleIntegratorArray" << endl;
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

//Integrator that uses atomic directive to store results
template<int N>
class SimpleIntegratorAtomic : public SimpleIntegrator<N>
{
    double sum;
public:
    SimpleIntegratorAtomic(const std::function<double(double)> &f_in, const size_t &size_in, const double &start_in, const double &end_in) : SimpleIntegrator<N>(f_in,size_in, start_in,end_in),
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

//Class to simplify integrator test running.
//Class will call to next one till 1
template< template<int> class Integrator, int N>
class IntegratorTest
{
    const std::function<double(double)> &f;
    size_t size;
    double begin;
    double end;
public:
    IntegratorTest(const std::function<double(double)> &f_in, const size_t &size_in, const double &start_in, const double &end_in) :
        f(f_in),
        size(size_in),
        begin(start_in),
        end(end_in)
    {


    }

    void run(map<int,double> &time_map)
    {
        //Create and run the current integrator
        Integrator<N> integrator{f,size,begin,end};
        double start = omp_get_wtime();
        double sum = integrator.run();
        time_map[N] += omp_get_wtime() - start;
        cout << "Pi (" << N << "): " << sum << endl;
        //create and call the next test
        IntegratorTest<Integrator,N-1> integrator_n{f,size,begin,end};
        integrator_n.run(time_map);
    }

};

//Class to simplify integrator test running.
//Class will call to next one, till here
template< template<int> class Integrator>
class IntegratorTest<Integrator,1>
{
    const std::function<double(double)> &f;
    size_t size;
    double begin;
    double end;
public:
    IntegratorTest(const std::function<int(int)> &f_in, const size_t &size_in, const double &start_in, const double &end_in) :
        f(f_in),
        size(size_in),
        begin(start_in),
        end(end_in)
    {


    }

    void run(map<int,double> &time_map)
    {
        //Create and run the current integrator
        Integrator<1> integrator{f,size,begin,end};
        double start = omp_get_wtime();
        double sum = integrator.run();
        time_map[1] += omp_get_wtime() - start;
        cout << "Pi (" << 1 << "): " << sum << endl;
        //it end here
    }

};

int main()
{
    map<int,double> time_map
    {
        {1,0.0f},
        {2,0.0f},
        {3,0.0f},
        {4,0.0f},
        {5,0.0f},
        {6,0.0f},
        {7,0.0f},
        {8,0.0f},
    };

    map<int,double> time_map_A
    {
        {1,0.0f},
        {2,0.0f},
        {3,0.0f},
        {4,0.0f},
        {5,0.0f},
        {6,0.0f},
        {7,0.0f},
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
        IntegratorTest<SimpleIntegratorAtomic,8> test{F,N,begin,end};
        test.run(time_map);
        IntegratorTest<SimpleIntegratorArray,8> testA{F,N,begin,end};
        testA.run(time_map_A);
    }


    cout << "Mean time [i]: " << "Atomic:" <<", "<< "Array: "<<" ms" << endl;

    for (int i = 1; i < 9; i++)
    {
        cout << "Mean time [" << i << "]: " << time_map[i]/(loop_total)*1000 <<", "<< time_map_A[i]/(loop_total)*1000 <<", "<< (time_map[i]/(loop_total)*1000 - time_map_A[i]/(loop_total)*1000)*100<< endl;
    }

    return 0;
}
