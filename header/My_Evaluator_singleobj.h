#pragma once

#include "nomad.hpp"
#include<vector>

#ifndef MY_EVALUATOR_SINGLEOBJ_H
#define MY_EVALUATOR_SINGLEOBJ_H

class My_Evaluator : public NOMAD::Evaluator
{
public:
	/*----------------------------------------*/
	/*               Constructor              */
	/*----------------------------------------*/
	My_Evaluator(const NOMAD::Parameters &p) : Evaluator(p) {}

	/*----------------------------------------*/
	/*               Destructor               */
	/*----------------------------------------*/
	~My_Evaluator(void) {}

	/*----------------------------------------*/
	/*               the problem              */
	/*----------------------------------------*/
	bool eval_x(NOMAD::Eval_Point &x, const NOMAD::Double &h_max, bool &count_eval) const;

	int req_index;
	double R_threshold; // reliability threshold
	std::vector<std::vector<double>> design_data, resiliance_th_data;

};

#endif // MY_EVALUATOR_SINGLEOBJ_H