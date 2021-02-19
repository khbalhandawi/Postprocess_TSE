#include "My_Evaluator.h"
#include "lookup_function.h"
#include "user_functions.h"

/*----------------------------------------------------*/
/*                         eval_x                     */
/*----------------------------------------------------*/
bool My_Evaluator::eval_x(NOMAD::Eval_Point &x, const NOMAD::Double &h_max, bool &count_eval) const {

	NOMAD::Double f, g1; // objective function

	int run_type = My_Evaluator::run_type;
	int obj_type = My_Evaluator::obj_type;
	std::vector<int> r_vec = My_Evaluator::req_vec; // get requirement vector
	std::vector<double> r_thresh = My_Evaluator::req_thresh; // get threshold vector
	std::vector<std::vector<double>> design_data = My_Evaluator::design_data; // get design data
	std::vector<std::vector<double>> resiliance_th_data = My_Evaluator::resiliance_th_data; // get requirement matrix (TH)
	std::vector<std::vector<double>> excess_th_data = My_Evaluator::excess_th_data; // get requirement matrix (TH)
	//--------------------------------------------------------------//
	// Read from data file

	count_eval = false;
	bool count_eval_lookup;
	std::vector<double> outputs, W_vector, R_vector, E_vector;

	int n_stages = 6;
	double W_vector_sum = 0.0, E_vector_sum = 0.0;
	std::vector<int> input_deposits; // extract different deposit types
	input_deposits.push_back(int(x[1].value())); // push back concept type
	int req_index;
	std::vector<double> resiliance_th, excess_th;

	if (x[0] == (x.size() - 2)) {
		for (size_t k = 0; k < (n_stages); ++k) {

			req_index = req_vec[k];
			resiliance_th = resiliance_th_data[req_index + 6];
			excess_th = excess_th_data[req_index + 6];

			if (k < (x.size() - 2)) {
				input_deposits.push_back(int(x[k + 2].value())); // get input vector
			}

			// remove -1 deposits from input:
			std::vector<int> lookup_vector;
			for (size_t m = 0; m < input_deposits.size(); ++m) {
				if (input_deposits[m] != -1) {
					lookup_vector.push_back(input_deposits[m]);
				}
			}

			outputs = lookup_function(lookup_vector, design_data, resiliance_th, excess_th);

			if (outputs[3] == 1.0) {
				W_vector.push_back(outputs[0]);
				R_vector.push_back(outputs[1]);
				E_vector.push_back(outputs[2]);
				count_eval_lookup = true;
				count_eval = true;

				//cout << outputs[0] << endl;
				//cout << outputs[1] << endl;

			}
			else if (outputs[3] == 0.0) {
				count_eval_lookup = false;
				count_eval = true;
				break;
			}

			W_vector_sum += W_vector.back();
			E_vector_sum += E_vector.back();
		}
	}

	//vector<double> Resiliance_constraint = { 0.4,0.1,0.1,0.1,0.1,0.99 };
	std::vector<double> Resiliance_constraint = r_thresh;

	if (count_eval_lookup) {
		if (obj_type == 0) { // set cumilative weight as objective
			x.set_bb_output(0, W_vector_sum);
		}
		else if (obj_type == 1) { // set cumilative excess as objective
			x.set_bb_output(0, E_vector_sum);
		}
		//x.set_bb_output(0, W_vector[5]);

		// assign constraint vector:
		std::vector<int> g_vector;
		NOMAD::Double g_n;
		for (size_t n = 0; n < Resiliance_constraint.size(); ++n) {
			g_n = Resiliance_constraint[n] - R_vector[n];
			x.set_bb_output(int(n + 1), g_n);
		}
		std::vector<int> output_log; // map input variable to output vector
		for (size_t k = 0; k < x.size(); ++k) {
			output_log.push_back(int(x[k].value())); // populate output vector
		}

		std::ofstream file("./MADS_output/mads_bb_calls.log", std::ofstream::app);
		writeTofile(output_log, &file);

	}
	else if (!count_eval_lookup) {
		x.set_bb_output(0, NOMAD::INF);
		x.set_bb_output(1, NOMAD::INF);
		x.set_bb_output(2, NOMAD::INF);
		x.set_bb_output(3, NOMAD::INF);
		x.set_bb_output(4, NOMAD::INF);
		x.set_bb_output(5, NOMAD::INF);
		x.set_bb_output(6, NOMAD::INF);

		std::vector<double> W_vector(n_stages, 0.0);

	}

	if (run_type == 1) { // if run type is an evaluation
		std::ofstream weight_file("./MADS_output/weight_design.log", std::ofstream::out);
		writeTofile_vector(W_vector, &weight_file);
		weight_file.close();

		std::ofstream excess_file("./MADS_output/excess_design.log", std::ofstream::out);
		writeTofile_vector(E_vector, &excess_file);
		weight_file.close();
	}

	return true;
}
