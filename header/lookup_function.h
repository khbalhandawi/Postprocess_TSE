#pragma once

#include "My_Evaluator.h"
#include "My_Extended_Poll.h"
#include <vector>

#ifndef LOOKUP_FUNCTION_H
#define LOOKUP_FUNCTION_H

/*-----------------------------------------------------------*/
/*               Lookup test file for results                */
/*-----------------------------------------------------------*/
std::vector<double> lookup_function(const std::vector<int> &input_deposits,
	const std::vector<std::vector<double>> &design_data,
	const std::vector<double> &resiliance_th,
	const std::vector<double> &excess_th);

/*-----------------------------------------------------------*/
/*       Function to check feasiblity of each design         */
/*-----------------------------------------------------------*/
std::vector<double> feasiblity_loop(const std::vector<std::vector<double>> &design_data, My_Evaluator *ev, My_Extended_Poll *ep);

#endif // LOOKUP_FUNCTION_H
