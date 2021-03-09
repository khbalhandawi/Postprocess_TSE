#include "lookup_function.h"
#include "nomad.hpp"

/*-----------------------------------------------------------*/
/*               Lookup test file for results                */
/*-----------------------------------------------------------*/
std::vector<double> lookup_function(const std::vector<int> &input_deposits,
	const std::vector<std::vector<double>> &design_data,
	const std::vector<double> &resiliance_th,
	const std::vector<double> &excess_th)
{
	double W, R, E, count_eval;

	std::vector<double> concept, i1, i2, i3, i4, i5, n_f_th, weight;

	concept = design_data[1];
	i1 = design_data[2];
	i2 = design_data[3];
	i3 = design_data[4];
	i4 = design_data[5];
	i5 = design_data[6];
	n_f_th = design_data[33];
	weight = design_data[35];

	//resiliance_th = design_data[45];
	// number of deposits:
	int value;

	std::vector<int> input;
	for (int i = 0; i < 6; ++i)
	{
		if (i < input_deposits.size()) {
			value = static_cast<int> (input_deposits[i]); // get input vector
		}
		else {
			value = -1;
		}
		input.push_back(value);
	}

	// number of branches
	size_t k = n_f_th.size();

	// Look up safety factor value
	std::vector<int> lookup;

	bool found = false;
	for (int i = 0; i < k; ++i)
	{
		lookup = { static_cast<int> (concept[i]), static_cast<int> (i1[i]),
			static_cast<int> (i2[i]), static_cast<int> (i3[i]),
			static_cast<int> (i4[i]), static_cast<int> (i5[i]) };

		if (input == lookup) {
			W = weight[i];
			R = resiliance_th[i];
			E = excess_th[i];
			count_eval = 1.0;
			found = true;
			// terminate the loop
			break;
		}
	}
	if (!found)
	{
		//cout << "objective: " << f << endl;
		count_eval = 0.0;
	}

	return { W, R, E, count_eval };

}

/*-----------------------------------------------------------*/
/*       Function to check feasiblity of each design         */
/*-----------------------------------------------------------*/
std::vector<double> feasiblity_loop(const std::vector<std::vector<double>> &design_data, My_Evaluator *ev, My_Extended_Poll *ep) 
{

	std::vector<double> concept, i1, i2, i3, i4, i5, n_f_th, weight;

	concept = design_data[1];
	i1 = design_data[2];
	i2 = design_data[3];
	i3 = design_data[4];
	i4 = design_data[5];
	i5 = design_data[6];
	n_f_th = design_data[33];
	weight = design_data[35];

	// number of branches
	size_t k = n_f_th.size();

	// Look up safety factor value
	std::vector<int> lookup;
	std::vector<double> feasiblity_vec;

	// loop over designs to check their feasiblity
	for (int i = 0; i < k; ++i) {

		lookup = { static_cast<int> (concept[i]), static_cast<int> (i1[i]),
			static_cast<int> (i2[i]), static_cast<int> (i3[i]),
			static_cast<int> (i4[i]), static_cast<int> (i5[i]) };

		NOMAD::Point xt(8);
		xt[0] = 6;
		for (size_t k = 0; k < 7; k++) {
			if (k < lookup.size()) {
				xt[k + 1] = lookup[k];
			}
			else {
				xt[k + 1] = -1;
			}
		}
		//cout << xt << endl;
		std::vector<NOMAD::Point> extended;
		ep->shuffle_padding(xt, &extended);

		bool feasible = false; // global feasiblity check
		// loop over variants until one of them satisfies requirement
		for (size_t k = 2; k < extended.size(); k++) {
			NOMAD::Eval_Point x_ex(extended[k].size(), 7);

			// map nomad point to eval point
			for (size_t j = 0; j < extended[k].size(); ++j) {
				x_ex[j] = extended[k][j];
			}
			NOMAD::Double hmax = 20.0;
			bool count = false;

			ev->eval_x(x_ex, hmax, count);
			//x_ex.display_eval(cout); // Debug evaluator

			// local feasibility check
			bool x_feasible = true;
			for (size_t i = 1; i < x_ex.get_bb_outputs().size(); ++i) {
				NOMAD::Double g_i = x_ex.get_bb_outputs()[i];
				if (g_i > 0.0) {
					x_feasible = false;
					break;
				}
			}

			// stop iterating and set global feasibility to true
			if (x_feasible) {
				feasible = true;
				break;
			}

		}

		// Output vector
		if (feasible) {
			feasiblity_vec.push_back(1.0);
		}
		else {
			feasiblity_vec.push_back(0.0);
		}
	}

	return feasiblity_vec;

}
