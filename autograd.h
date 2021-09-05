#ifndef AUTOGRAD_H
#define AUTOGRAD_H
#include "utils.h"

using Eigen::MatrixX;

template <typename dtype>
struct graph_node;
template <typename dtype>
class Einsor;

template<typename dtype>
inline void calgrad_add(Einsor<dtype>* a, Einsor<dtype>* b, Einsor<dtype>* result)
{
	if (a->require_grad)
		*a->grad += *result->grad;
	if (b->require_grad)
		*b->grad += *result->grad;
}

template<typename dtype>
inline void calgrad_sub(Einsor<dtype>* a, Einsor<dtype>* b, Einsor<dtype>* result)
{
	if (a->require_grad)
		*a->grad += *result->grad;
	if (b->require_grad)
		*b->grad -= *result->grad;
}

template<typename dtype>
inline void calgrad_prod(Einsor<dtype>* a, Einsor<dtype>* b, Einsor<dtype>* result)
{
	if (a->require_grad)
		*a->grad += *result->grad * b->data->transpose();
	if (b->require_grad)
		*b->grad += a->data->transpose() * *result->grad;
}

template<typename dtype>
inline void calgrad_sum(Einsor<dtype>* a, Einsor<dtype>* b, Einsor<dtype>* result)
{
	*a->grad += MatrixX<dtype>::Constant(a->rows(), a->cols(), 1);
}

template<typename dtype>
inline void calgrad_div(Einsor<dtype>* a, Einsor<dtype>* b, Einsor<dtype>* result)
{
	Eigen::MatrixX<dtype> denominator = 1. / b->data->array();
	Eigen::MatrixX<dtype> grad_a = result->grad->array() * denominator.array();
	if (a->require_grad)
		*a->grad += grad_a;
	if (b->require_grad)
		*b->grad -= (a->data->array() * grad_a.array() * denominator.array()).matrix();
}

template<typename dtype>
inline void calgrad_hadamardprod(Einsor<dtype> *a, Einsor<dtype> *b, Einsor<dtype> *result)
{
	if(a->require_grad)
		*a->grad += (b->data->array() * result->grad->array()).matrix();
	if(b->require_grad)
		*b->grad += (a->data->array() * result->grad->array()).matrix();
}

template<typename dtype>
Einsor<dtype>& hadamardprod(Einsor<dtype> &a, Einsor<dtype> &b)
{
	a.root = a.require_grad ? a.root : b.root;
	auto result = a.makeDynamicEinsor(new Einsor<dtype>(a.data->array() * b.data->array(), a.require_grad || b.require_grad));
	if(result->require_grad)
		result->back_nodes = make_shared<graph_node<dtype>>(&a, &b, &calgrad_hadamardprod<dtype>);
	return *result;
}

template<typename dtype>
class Einsor
{
    using Matrix = MatrixX<dtype>;
    using node = graph_node<dtype>;
public:
	Einsor(){}
	~Einsor(){}
    int rows()
    {
        if (this->data != nullptr)
            return data->rows();
        else return -1;        
    }
    int cols()
    {
        if (this->data != nullptr)
            return data->cols();
        else return -1;        
    }
    auto operator << (const dtype &scalar)
    {
        return data->operator<<(scalar);
    }
    friend std::ostream& operator << (std::ostream& printout, Einsor<dtype>& e)
    {
        printout << "[data:\n";
        if (e.data != nullptr)
            printout << *(e.data);
        if (e.grad != nullptr)
        {
            printout << "\ngrad:\n" << *(e.grad); 
        }
        printout << "]\n";
        return printout;
    }
	Einsor(int rows, int cols, bool require_grad=false)
	{
        data = make_shared<Matrix>(rows, cols);
		set_grad(require_grad);
	}
	Einsor(int rows, int cols, bool require_grad, dtype constant)
	{
		data = make_shared<Matrix>(Matrix::Constant(rows, cols, constant));
		set_grad(require_grad);
	}
	Einsor(Matrix m, bool require_grad=false)
	{
        data = make_shared<Matrix>(m);
		set_grad(require_grad);
	}
    Einsor(dtype scalar, bool require_grad=false)
    {
        data = make_shared<Matrix>(Matrix::Constant(1, 1, scalar));
        set_grad(require_grad);
    }
	
	void set_grad(bool _require_grad)
	{
		require_grad = _require_grad;
		if (require_grad)
			grad = make_shared<Matrix>(Eigen::MatrixXd::Zero(this->rows(), this->cols()));
	}

	void grad_zero()
	{
		if (require_grad)
			grad = make_shared<Matrix>(Eigen::MatrixXd::Zero(this->rows(), this->cols()));
	}

    void replaceDuplicateNode(Einsor<dtype>* oldptr, shared_ptr<Einsor<dtype>> newptr)
    {
        if (back_nodes != nullptr)
        {
            if(back_nodes->first_node == oldptr)
                back_nodes->first_node = newptr.get();
            if(back_nodes->first_node != nullptr)
                back_nodes->first_node->replaceDuplicateNode(oldptr, newptr);
			if(back_nodes->sec_node == oldptr)
                back_nodes->sec_node = newptr.get();
            if(back_nodes->sec_node != nullptr)
                back_nodes->sec_node->replaceDuplicateNode(oldptr, newptr);
        }
    }

	template<typename ... Args>
	shared_ptr<Einsor<dtype>> makeDynamicEinsor(Args ... args)
	{
		shared_ptr<Einsor<dtype>> result(args...);
		
		result->root = root;
		root->dynamic_node_set.insert(result);
		
		return result;
	}

	Einsor<dtype>& operator + (Einsor<dtype>& m)
	{
		auto result = makeDynamicEinsor(new Einsor<dtype>(data->operator+(*m.data), require_grad || m.require_grad));
		if (result->require_grad)
            result->back_nodes = make_shared<node>(this, &m, &calgrad_add<dtype>);
		return *result;
	}

	Einsor<dtype>& operator - (Einsor<dtype>& m)
	{
		auto result = makeDynamicEinsor(new Einsor<dtype>(data->operator-(*m.data), require_grad || m.require_grad));
		if (result->require_grad)
            result->back_nodes = make_shared<node>(this, &m, &calgrad_sub<dtype>);
		return *result;
	}

	Einsor<dtype>& operator * (Einsor<dtype>& m)
	{
		// root = require_grad ? root : m.root;
		auto result = makeDynamicEinsor(new Einsor<dtype>(data->operator*(*m.data), require_grad || m.require_grad));
		if (result->require_grad)
			result->back_nodes = make_shared<node>(this, &m, &calgrad_prod<dtype>);
		return *result;
	}

	Einsor<dtype>& operator / (Einsor<dtype>& m)
	{
		auto result = makeDynamicEinsor(new Einsor<dtype>(data->array().operator/(m.data->array()), require_grad || m.require_grad));
		if (result->require_grad)
			result->back_nodes = make_shared<node>(this, &m, &calgrad_div<dtype>);
		return *result;
	}

	void copyFrom(Einsor<dtype>& m)
	{
		data = m.data;
		grad = m.grad;
		back_nodes = m.back_nodes;
		require_grad = m.require_grad;
	}

	Einsor<dtype>& operator = (Einsor<dtype>& m)
	{
        if(require_grad)
			m.replaceDuplicateNode(this, makeDynamicEinsor(new Einsor<dtype>(*this)));
		copyFrom(m);
		return *this;
	}

	Einsor<dtype>& operator += (Einsor<dtype>& m)
	{
		require_grad = require_grad || m.require_grad;
		if (require_grad)
			this->back_nodes = make_shared<node>(makeDynamicEinsor(new Einsor<dtype>(*this)).get(), &m, &calgrad_add<dtype>);
		*this->data += *m.data;
		return *this;
	}

	Einsor<dtype>& sum()
	{
        double dt = data->array().sum();
        auto result = makeDynamicEinsor(new Einsor<dtype>(dt, require_grad));
        if (require_grad)
            result->back_nodes = make_shared<node>(this, nullptr, &calgrad_sum<dtype>);
        return *result;
	}

	void backward()
	{
		if (require_grad && back_nodes != nullptr)
		{
			(*back_nodes->grad_func)(back_nodes->first_node, back_nodes->sec_node, this);
			if (back_nodes->first_node != nullptr)
				back_nodes->first_node->backward();
			if (back_nodes->sec_node != nullptr)
				back_nodes->sec_node->backward();
		}
	}
    shared_ptr<node> back_nodes = nullptr;
    shared_ptr<Matrix> data = nullptr;
    shared_ptr<Matrix> grad = nullptr;
	bool require_grad = false;

	// "root" is not pointing to the true root of the computational graph
	// but the first node which is not temporary.
	// meanful only when calling makeDynamicEinsor to get temporary.
	Einsor<dtype> *root = this;
	// store shared_ptr of temporary nodes incase they are needed for backward.
	set<shared_ptr<Einsor<dtype>>> dynamic_node_set;
protected:
private:
};
using EinsorD = Einsor<double>;

template <typename dtype>
struct graph_node
{
public:
	Einsor<dtype> *first_node;
	Einsor<dtype> *sec_node;
	void (*grad_func) (Einsor<dtype> *a, Einsor<dtype> *b, Einsor<dtype> *result);
	graph_node(Einsor<dtype> * a, Einsor<dtype> *b, void (*func)(Einsor<dtype> *a, Einsor<dtype> *b, Einsor<dtype> *result)): 
																				first_node(a), sec_node(b), grad_func(func){}
};

void testEinsor();

#endif