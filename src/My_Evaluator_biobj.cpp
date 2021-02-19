
#include "My_Evaluator_biobj.h"
#include "user_functions.h"

#include <vector>

/*----------------------------------------------------*/
/*                         eval_x                     */
/*----------------------------------------------------*/
bool My_Evaluator::eval_x(NOMAD::Eval_Point   & x,
	const NOMAD::Double & h_max,
	bool         & count_eval) const
{
	NOMAD::Double f1; // objective function
	NOMAD::Double f2; // objective function

	int req_index = My_Evaluator::req_index;
	std::vector<std::vector<double>> design_data = My_Evaluator::design_data; // get design data
	std::vector<std::vector<double>> resiliance_ip_data = My_Evaluator::resiliance_ip_data; // get requirement matrix (IP)
	std::vector<std::vector<double>> resiliance_th_data = My_Evaluator::resiliance_th_data; // get requirement matrix (TH)
	//--------------------------------------------------------------//
	// Read from data file

	count_eval = false;

	std::vector<double> concept, i1, i2, i3, i4, i5, n_f_th, V_c, weight, attribute;
	concept = design_data[1];
	i1 = design_data[2];
	i2 = design_data[3];
	i3 = design_data[4];
	i4 = design_data[5];
	i5 = design_data[6];
	n_f_th = design_data[33];
	V_c = design_data[59];
	weight = design_data[35];

	if (req_index == 0) {
		attribute = V_c; //volume of capability
	}
	else {
		attribute = resiliance_th_data[(6 + req_index)]; //req_index_i
	}


	// number of deposits:
	int n = static_cast<int> (x[0].value());
	int value;

	std::vector<int> input;
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
	std::vector<int> lookup;

	bool found = false;
	for (int i = 0; i < k; ++i)
	{
		lookup = { static_cast<int> (concept[i]), static_cast<int> (i1[i]),
			static_cast<int> (i2[i]), static_cast<int> (i3[i]),
			static_cast<int> (i4[i]), static_cast<int> (i5[i]) };

		if (input == lookup) {
			f1 = -attribute[i];
			f2 = weight[i];
			x.set_bb_output(0, f1);
			x.set_bb_output(1, f2);
			count_eval = true;
			found = true;

			ofstream file("./MADS_output/mads_bb_calls_biobj.log", ofstream::app);
			writeTofile(input, &file);

			//std::cout << "objective: " << f << std::endl;
			// terminate the loop
			break;
		}
	}
	if (!found)
	{
		x.set_bb_output(0, NOMAD::INF);
		x.set_bb_output(1, NOMAD::INF);
		//std::cout << "objective: " << f << std::endl;
		count_eval = false;
	}
	return true;
}