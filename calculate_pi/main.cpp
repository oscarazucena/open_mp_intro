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
    const double getRange()
    {
        return end-start;
    }
public:
    SimpleIntegrator(const std::function<double(double)> &f_in, const size_t &size_in, const double &start_in, const double &end_in) :
        f(f_in),
        size(size_in),
        start(start_in),
        end(end_in)
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
    }
    double run() override
    {
        omp_set_num_threads(N);
        double temp_del = this->getRange()/this->size;
#pragma omp parallel
        {
            double sum_temp = 0.0;
            int id = omp_get_thread_num();
            int c_start = this->size*id/N;
            int c_end = this->size*(id+1)/N;
            double x_temp =  c_start*temp_del + temp_del/2;

            for(; c_start < c_end; c_start++)
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
            int c_start = this->size*id/N;
            int c_end = this->size*(id+1)/N;
            double x_temp =  c_start*temp_del + temp_del/2;

            for(; c_start < c_end; c_start++)
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

//Integrator that uses atomic directive to store results
template<int N>
class SimpleIntegratorForLoop : public SimpleIntegrator<N>
{
    double sum;
public:
    SimpleIntegratorForLoop(const std::function<double(double)> &f_in, const size_t &size_in, const double &start_in, const double &end_in) : SimpleIntegrator<N>(f_in,size_in, start_in,end_in)
    {

    }
    double run() override
    {
        omp_set_num_threads(N);
        double temp_del = this->getRange()/this->size;
        double sum_temp = 0.0f;
        double temp_del_2 = temp_del/2.0f;
        int c = 0;
        double x_temp = temp_del_2;
#pragma omp parallel for reduction(+:sum_temp,x_temp)
        for(c=0; c < this->size; c++)
        {
            x_temp += temp_del;
            sum_temp += 4.0/(1.0 + x_temp*x_temp);
        }
        return sum_temp*temp_del;
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
        //cout << "Pi (" << N << "): " << sum << endl;
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
        //cout << "Pi (" << 1 << "): " << sum << endl;
        //it ends here
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

    map<int,double> time_map_for
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
        IntegratorTest<SimpleIntegratorAtomic,8> test_array{F,N,begin,end};
        test_array.run(time_map);
        IntegratorTest<SimpleIntegratorArray,8> test_atomic{F,N,begin,end};
        test_atomic.run(time_map_A);
        IntegratorTest<SimpleIntegratorForLoop,8> test_for{F,N,begin,end};
        test_for.run(time_map_for);
    }

    auto calculate = [loop_total=loop_total](const int i, const map<int,double> &time_m) -> double
    {
        const auto iter = time_m.find(i);
        if (iter != time_m.end())
        {
            return iter->second/(loop_total)*1000.0f;
        }
        return 0.0f;
    };

    cout << "Mean time [i]: " << "Atomic:" <<", "<< "Array: "<<"For Loop: " << endl;

    for (int i = 1; i < 9; i++)
    {
        cout << "Mean time [" << i << "]: " << calculate(i,time_map) <<", "<< calculate(i,time_map_A) <<", "<< calculate(i,time_map_for)<< endl;
    }

    return 0;
}
