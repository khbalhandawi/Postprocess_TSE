
#include "My_Evaluator_singleobj.h"
#include "user_functions.h"

#include <vector>

/*----------------------------------------------------*/
/*                         eval_x                     */
/*----------------------------------------------------*/
bool My_Evaluator::eval_x(NOMAD::Eval_Point   & x,
	const NOMAD::Double & h_max,
	bool         & count_eval) const
{
	NOMAD::Double f, g1; // objective function

	int req_index = My_Evaluator::req_index;
	double R_threshold = My_Evaluator::R_threshold; // reliability threshold
	vector<vector<double>> design_data = My_Evaluator::design_data; // get design data
	vector<vector<double>> resiliance_th_data = My_Evaluator::resiliance_th_data; // get requirement matrix (TH)
	//--------------------------------------------------------------//
	// Read from data file

	count_eval = false;

	vector<double> concept, i1, i2, i3, i4, i5, n_f_th, V_c, weight, R_nominal, E_nominal, objective, constraint;
	concept = design_data[1];
	i1 = design_data[2];
	i2 = design_data[3];
	i3 = design_data[4];
	i4 = design_data[5];
	i5 = design_data[6];
	n_f_th = design_data[33];
	V_c = design_data[59];
	weight = design_data[35];
	R_nominal = design_data[55]; // resiliance_th_uni
	E_nominal = design_data[63]; // excess_th_uni


	if (req_index == 0) {
		constraint = R_nominal; //volume of capability
	}
	else {
		constraint = resiliance_th_data[(6 + req_index)]; //req_index_i
	}

	objective = E_nominal;

	// number of deposits:
	int n = static_cast<int> (x[0].value());
	int value;

	vector<int> input;
	for (int i = 0; i < 6; ++i)
	{
		if (i < (n + 1)) {
			value = static_cast<int> (x[i + 1].value()); // get input vector
		}
		else {
			value = -1;
		}
		input.push_back(value);
	}

	// number of branches
	size_t k = n_f_th.size();

	// Look up safety factor value
	vector<int> lookup;

	bool found = false;
	for (int i = 0; i < k; ++i)
	{
		lookup = { static_cast<int> (concept[i]), static_cast<int> (i1[i]),
			static_cast<int> (i2[i]), static_cast<int> (i3[i]),
			static_cast<int> (i4[i]), static_cast<int> (i5[i]) };

		if (input == lookup) {
			f = objective[i];
			g1 = R_threshold - constraint[i];
			x.set_bb_output(0, g1);
			x.set_bb_output(1, f);
			count_eval = true;
			found = true;

			ofstream file("./MADS_output/mads_bb_calls.log", ofstream::app);
			writeTofile(input, &file);

			//cout << "objective: " << f << endl;
			// terminate the loop
			break;
		}
	}
	if (!found)
	{
		x.set_bb_output(0, NOMAD::INF);
		x.set_bb_output(1, NOMAD::INF);
		//cout << "objective: " << f << endl;
		count_eval = false;
	}
	return true;
}