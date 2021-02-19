#pragma once

#include "nomad.hpp"
#include<vector>

#ifndef MY_EVALUATOR_BIOBJ_H
#define MY_EVALUATOR_BIOBJ_H

class My_Evaluator : public NOMAD::Multi_Obj_Evaluator
{
public:
	/*----------------------------------------*/
	/*               Constructor              */
	/*----------------------------------------*/
	My_Evaluator(const NOMAD::Parameters &p) : Multi_Obj_Evaluator(p) {}

	/*----------------------------------------*/
	/*               Destructor               */
	/*----------------------------------------*/
	~My_Evaluator(void) {}

	/*----------------------------------------*/
	/*               the problem              */
	/*----------------------------------------*/
	bool eval_x(NOMAD::Eval_Point &x, const NOMAD::Double & h_max, bool &count_eval) const;

	int req_index;
	std::vector<std::vector<double>> design_data, resiliance_ip_data, resiliance_th_data;

};

#endif // MY_EVALUATOR_BIOBJ_H