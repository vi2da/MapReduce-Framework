// vi2da

#include <iostream>
#include <semaphore.h>
#include <algorithm>
#include <atomic>
#include "MapReduceFramework.h"
#include "Barrier.h"



/**
 * a class that holds all the context variables and containers
 */
class generalContext{
private:
    const InputVec* _inputVec;
    size_t _next_pair_to_read;
    int _no_more_jobs_to_reduce;
    std::vector<IntermediateVec*> _middleVec;


    OutputVec* _outputVec;


    sem_t* _shuffle_semaphore;
    pthread_mutex_t* _inputVec_mutex;
    pthread_mutex_t* _middleVec_mutex;
    pthread_mutex_t* _outputVec_mutex;
    pthread_mutex_t* _read_sem_val_mutex;


public:

    bool _shuffle_is_finished;
    const MapReduceClient* _client;

    Barrier* _barrier;
    generalContext(const MapReduceClient* client, Barrier* barrier,
                   const InputVec* inputVec, OutputVec* outputVec, sem_t* my_semaphore,
                   pthread_mutex_t* inputVec_mutex, pthread_mutex_t* middleVec_mutex, pthread_mutex_t* outputVec_mutex,
                   pthread_mutex_t* read_sem_val_mutex);
    const InputPair* get_pair_for_map_work();
    void add_to_middleVec(IntermediateVec* vec_to_push);
    IntermediateVec* get_vec_for_reduce_work();
    void add_to_OutputVec(K3* key, V3* value);

};


/**
 * the general context class constructor
 * @param client the MapReduce client that holds the Map and Reduce functions
 * @param barrier a barrier object
 * @param inputVec the input vector for the framework to work with
 * @param outputVec the output vector to which the framework writes its results
 * @param numThreads the number of threads the framework works with, including main thread
 * @param my_semaphore a semaphore object to manage the number of jobs
 * @param inputVec_mutex a mutex object to protect the input vector
 * @param middleVec_mutex a mutex object to protect the vector the Shuffle writes to
 * @param outputVec_mutex a mutex object to protect the output vector
 * @param read_sem_val_mutex a mutex object to protect the semaphore count
 */
generalContext::generalContext(const MapReduceClient *client, Barrier *barrier, const InputVec *inputVec,
                               OutputVec *outputVec, sem_t *my_semaphore,
                               pthread_mutex_t *inputVec_mutex, pthread_mutex_t *middleVec_mutex,
                               pthread_mutex_t *outputVec_mutex, pthread_mutex_t *read_sem_val_mutex)
{
    _shuffle_is_finished = false;
    _shuffle_semaphore = my_semaphore;
    _inputVec_mutex = inputVec_mutex;
    _middleVec_mutex = middleVec_mutex;
    _outputVec_mutex = outputVec_mutex;
    _read_sem_val_mutex = read_sem_val_mutex;

    _client = client;
    _inputVec = inputVec;
    _next_pair_to_read = 0;
    _no_more_jobs_to_reduce = 0;
    _outputVec = outputVec;
    _barrier = barrier;
}


/**
 * gets an InputPair pair from the input vector for the thread
 * to work on in the map phase
 * @return a pointer to InputPair pair from the input vector if succeeded,
 * nullptr otherwise
 */
const InputPair* generalContext::get_pair_for_map_work()
{
    const InputPair* to_return = nullptr;
    if(pthread_mutex_lock(_inputVec_mutex))
    {
        std::cerr << "pthread_mutex_lock failed"<< std::endl;
        exit(1);
    }

    if( _next_pair_to_read < _inputVec->size())
    {
        to_return = &(_inputVec->at(_next_pair_to_read));
        _next_pair_to_read++;
    }

    if(pthread_mutex_unlock(_inputVec_mutex))
    {
        std::cerr << "pthread_mutex_unlock failed"<< std::endl;
        exit(1);
    }
    return to_return;
}


/**
 * a function for the Shuffle to call in order to push a vector
 * of all max pairs it collected from all the threads' containers
 * @param vec_to_push the vector to push to the middle vector (priority queue)
 */
void generalContext::add_to_middleVec(IntermediateVec* vec_to_push)
{
    if(pthread_mutex_lock(_middleVec_mutex))
    {
        std::cerr << "pthread_mutex_lock failed"<< std::endl;
        exit(1);
    }

    // adding new vector of <k2, v2> pairs with the same k2 in all pairs
    _middleVec.push_back(vec_to_push);

    if(pthread_mutex_unlock(_middleVec_mutex))
    {
        std::cerr << "pthread_mutex_unlock failed"<< std::endl;
        exit(1);
    }

    // adding +1 to the sem count to wake up a reduce thread
    if(sem_post(_shuffle_semaphore))
    {
        std::cerr << "sem_post failed"<< std::endl;
        exit(1);
    }
}


/**
 * gets an IntermediateVec vector from the middleVector for the calling thread
 * to work on in the reduce phase
 * @return a pointer to IntermediateVec vector from the middleVector if succeeded,
 * nullptr otherwise
 */
IntermediateVec* generalContext::get_vec_for_reduce_work()
{
    IntermediateVec* to_return = nullptr;

    if(pthread_mutex_lock(_read_sem_val_mutex))
    {
        std::cerr << "1pthread_mutex_lock failed"<< std::endl;
        exit(1);
    }
    bool midlevec_is_empty = _middleVec.empty();

    if(_shuffle_is_finished && midlevec_is_empty)
    {
        _no_more_jobs_to_reduce = 1;
        if (sem_post(_shuffle_semaphore))
        {
            std::cerr << "sem_post failed" << std::endl;
            exit(1);
        }

        if(pthread_mutex_unlock(_read_sem_val_mutex))
        {
            std::cerr << "pthread_mutex_unlock failed"<< std::endl;
            exit(1);
        }
        return to_return;
    }
    if(pthread_mutex_unlock(_read_sem_val_mutex))
    {
        std::cerr << "pthread_mutex_unlock failed"<< std::endl;
        exit(1);
    }

    if(sem_wait(_shuffle_semaphore))
    {
        std::cerr << "sem_wait failed" << std::endl;
        exit(1);
    }

    if(pthread_mutex_lock(_middleVec_mutex))
    {
        std::cerr << "2pthread_mutex_lock failed"<< std::endl;
        exit(1);
    }
    if(_no_more_jobs_to_reduce)
    {
        if(sem_post(_shuffle_semaphore))
        {
            std::cerr << "sem_post failed"<< std::endl;
            exit(1);
        }
    }
    else
    {
        // if we reched here it means that there is a job to take
        to_return = _middleVec.back();
        _middleVec.pop_back();
    }

    if(pthread_mutex_unlock(_middleVec_mutex))
    {
        std::cerr << "pthread_mutex_unlock failed"<< std::endl;
        exit(1);
    }

    return to_return;
}


/**
 * adds a pair of <K3* key, V3*> to the output vector
 * @param key the key of the pair to add
 * @param value the value of the pair to add
 */
void generalContext::add_to_OutputVec(K3* key, V3* value)
{
    if(pthread_mutex_lock(_outputVec_mutex))
    {
        std::cerr << "pthread_mutex_lock failed"<< std::endl;
        exit(1);
    }

    _outputVec->push_back({key, value});

    if(pthread_mutex_unlock(_outputVec_mutex))
    {
        std::cerr << "pthread_mutex_unlock failed"<< std::endl;
        exit(1);
    }
}


/**
 * a context for a single thread worker, containing its
 * container and relevant functions for the thread
 */
class WorkerContext {
private:
    IntermediateVec _map_container;
public:
    generalContext* _general_context;

    WorkerContext();
    void push_to_container(K2* key, V2* value);
    IntermediatePair pop_from_container();
    IntermediatePair get_max_pair();

    void sort_my_container();
};


/**
 * a constructor for the WorkerContext class
 */
WorkerContext::WorkerContext()
{
    _general_context = nullptr;
}


/**
 * gets a pointer to the maximal pair (pair with max key)
 * in the threads container
 * @return pointer to the maximal pair in the threads container,
 * nullptr if failed
 */
IntermediatePair WorkerContext::get_max_pair()
{
    IntermediatePair to_return = {nullptr, nullptr};
    if(!_map_container.empty())
    {
        to_return.first = _map_container.back().first;
        to_return.second = _map_container.back().second;
    }
    return to_return;
}


/**
 * pops the top elment from the calling thread's container
 * and returns it
 * @return the top elment from the calling thread's container
 */
IntermediatePair WorkerContext::pop_from_container()
{
    IntermediatePair to_return =  {_map_container.back().first, _map_container.back().second};
    _map_container.pop_back();
    return to_return;
}


/**
 * adss a pair of <K2*, V2*> to the container of the thread.
 * @param key the key of the pair of add
 * @param value the value of the pair to add
 */
void WorkerContext::push_to_container(K2 *key, V2 *value)
{
    _map_container.push_back({key, value});
}


/**
 * a comparator for the IntermediatePair pairs
 * @param first the first element in the pair
 * @param second the second element in the pair
 * @return 1 if first < second, 0 otherwise
 */
bool compareK2(IntermediatePair first, IntermediatePair second)
{
    // delegates the comparison to the user's implementation of k2::<operator
    return  *first.first < *second.first;
}


/**
 * sorts the contaier of the calling thread, by the keys of the
 * pairs in the container
 */
void WorkerContext::sort_my_container()
{
    std::sort(_map_container.begin(), _map_container.end(),compareK2);
}


/**
 * the map function for the threads to run in the map phase
 * @param worker_context the context of the thread that runs the map fuction
 */
void exec_map(WorkerContext* worker_context)
{
    int still_have_jobs = 1;
    while(still_have_jobs)
    {
        const InputPair* curr_pair = worker_context->_general_context->get_pair_for_map_work();
        if(curr_pair != nullptr)
        {
            worker_context->_general_context->_client->map(curr_pair->first, curr_pair->second, worker_context);
        }
        else
        {
            still_have_jobs = 0;
        }
    }

    worker_context->sort_my_container();

    // here we activate the barrier and we will pass this point only when all other threads
    // finished their exec_map phase
    worker_context->_general_context->_barrier->barrier();
}


/**
 * the shuffle function. called only by thread 0, who is designated to run
 * the shuffle phase
 * @param worker_context_arr an array of all the worker threads contexts
 * @param arr_size the size of the worker_context array
 */
void exec_shuffle(WorkerContext* worker_context_arr, int arr_size)
{
    while(true)
    {
        // in this loop we find the max K2 key in all containers
        IntermediatePair max_key_pair_p = {nullptr, nullptr};


        for(int i = 0; i < arr_size; i++)
        {

            IntermediatePair curr_key_pair = worker_context_arr[i].get_max_pair();;
            if(max_key_pair_p.first == nullptr && max_key_pair_p.second == nullptr)
            {
                max_key_pair_p.first = curr_key_pair.first;
                max_key_pair_p.second = curr_key_pair.second;
            }
            else
            {
                if(!(curr_key_pair.first == nullptr && curr_key_pair.second == nullptr))
                {
                    if(*max_key_pair_p.first < *curr_key_pair.first)
                    {
                        max_key_pair_p.first = curr_key_pair.first;
                        max_key_pair_p.second = curr_key_pair.second;
                    }
                }
            }
        }

        // if all containers are empty we exit the function
        if((max_key_pair_p.first == nullptr) && (max_key_pair_p.second == nullptr))
        {
            worker_context_arr[0]._general_context->_shuffle_is_finished = true;
            return;
        }

        IntermediateVec* same_key_vec = new std::vector<IntermediatePair>();

        for(int i=0; i<arr_size; i++)
        {
            WorkerContext* curr_context = worker_context_arr + i;
            IntermediatePair curr_key_pair = curr_context->get_max_pair();

            while(!(curr_key_pair.first == nullptr && curr_key_pair.second == nullptr)
                  && !(*curr_key_pair.first < *max_key_pair_p.first)
                  && !(*max_key_pair_p.first < *curr_key_pair.first))
            {
                same_key_vec->push_back(curr_context->pop_from_container());
                curr_key_pair.first = curr_context->get_max_pair().first;
                curr_key_pair.second = curr_context->get_max_pair().second;
            }
        }

        worker_context_arr[0]._general_context->add_to_middleVec(same_key_vec);
    }

}


/**
 * the reduce function to run by all the threads in the reduce phase
 * @param worker_context the context of the calling thread
 */
void exec_reduce(WorkerContext* worker_context)
{
    while(true)
    {
        IntermediateVec* curr_vec_to_reduce = worker_context->_general_context->get_vec_for_reduce_work();
        if(curr_vec_to_reduce == nullptr)
        {
            //std::cerr <<  "reduce phase is fhinished" << std::endl;
            return;
        }
        worker_context->_general_context->_client->reduce(curr_vec_to_reduce, worker_context);
        delete curr_vec_to_reduce;
    }
}


/**
 * the function each thread runs upon initialization
 * @param arg the arg parameter
 * @return pointer of type void*
 */
void* worker_thread_func(void* arg)
{
    WorkerContext* tc = (WorkerContext*) arg;
    exec_map(tc);

    tc->_general_context->_barrier->barrier();

    // when we arrive here, the map phase finished and the container is sorted.
    // now the thread waits at the barrier until all the other threads finish their exec_map phase
    exec_reduce(tc);

    return nullptr;
}


/**
 * adds a new IntermediatePair to the container of the calling thread.
 * this function is called by the Map function.
 * @param key the key of the pair to add
 * @param value the value of the pair to add
 * @param context pointer to the calling thread context
 */
void emit2 (K2* key, V2* value, void* context)
{
    WorkerContext* tc = (WorkerContext*) context;
    tc->push_to_container(key, value);
}


/**
 * adds a new OutputPair to the output vector
 * this function is called by the Reduce function.
 * @param key the key of the pair to add
 * @param value the value of the pair to add
 * @param context pointer to the calling thread context
 */
void emit3 (K3* key, V3* value, void* context)
{
    WorkerContext* tc = (WorkerContext*) context;
    tc->_general_context->add_to_OutputVec(key, value);
}


/**
 * the framework that manages the Map Reduce functionality
 * @param client a client class implementation, with its Map and Reduce implementation
 * @param inputVec the input vector for the framework to process
 * @param outputVec the output vector to which the framework puts its processed results
 * @param multiThreadLevel the number of worker threads to work with, including main thread
 */
void runMapReduceFramework(const MapReduceClient& client,
                           const InputVec& inputVec, OutputVec& outputVec,
                           int multiThreadLevel)
{
    Barrier barrier(multiThreadLevel);
    sem_t shuffle_semaphore;
    if (sem_init(&shuffle_semaphore, 0, 0)) {
        std::cerr << "sem_init failed" << std::endl;
        exit(1);
    }
    pthread_mutex_t inputVec_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t middleVec_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t outputVec_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t read_sem_val_mutex = PTHREAD_MUTEX_INITIALIZER;


    // initializing the context for this execution of the framework
    generalContext framework_context = generalContext(&client, &barrier, &inputVec, &outputVec,
                                                      &shuffle_semaphore, &inputVec_mutex, &middleVec_mutex,
                                                      &outputVec_mutex, &read_sem_val_mutex);


    pthread_t threads[multiThreadLevel - 1];
    WorkerContext contexts[multiThreadLevel];

    for (int i = 0; i < multiThreadLevel; i++) {
        contexts[i]._general_context = &framework_context;
    }


    for (int i = 0; i < multiThreadLevel - 1; ++i) {
        pthread_create(threads + i, nullptr, worker_thread_func, contexts + i);
    }

    WorkerContext *main_t_context = contexts + (multiThreadLevel - 1);

    // making the main thread (thread 0) run the exec_map, exec_shuffle and exec_reduce functions
    exec_map(main_t_context);

    contexts[0]._general_context->_barrier->barrier();

    exec_shuffle(contexts, multiThreadLevel);

    exec_reduce(main_t_context);

    for (int i = 0; i < multiThreadLevel; i++) {
        if (sem_post(&shuffle_semaphore)) {
            std::cerr << "sem_post failed" << std::endl;
            exit(1);
        }
    }

    for (int j = 0; j < multiThreadLevel - 1; j++) {
        if (pthread_join(threads[j], nullptr)) {
            std::cerr << "pthread_join failed" << std::endl;
            exit(1);
        }
    }

    if (pthread_mutex_destroy(&inputVec_mutex)) {
        fprintf(stderr, "pthread_mutex_destroy failed.");
        exit(1);
    }
    if (pthread_mutex_destroy(&middleVec_mutex)) {
        fprintf(stderr, "pthread_mutex_destroy failed.");
        exit(1);
    }
    if (pthread_mutex_destroy(&outputVec_mutex)) {
        fprintf(stderr, "pthread_mutex_destroy failed.");
        exit(1);
    }
    if (pthread_mutex_destroy(&read_sem_val_mutex)) {
        fprintf(stderr, "pthread_mutex_destroy failed.");
        exit(1);
    }
    if (sem_destroy(&shuffle_semaphore)) {
        fprintf(stderr, "sem_destroy failed.");
        exit(1);
    }
}