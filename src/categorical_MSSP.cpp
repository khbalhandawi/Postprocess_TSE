/*-------------------------------------------------------------------*/
/*            Example of a problem with categorical variables        */
/*-------------------------------------------------------------------*/
/*                                                                   */
/*  . portfolio problem with 3 assets                                */
/*                                                                   */
/*  . NOMAD is used in library mode                                  */
/*                                                                   */
/*  . the number of variables can be 3,5, or 7, depending on the     */
/*    number of assets considered in the portfolio                   */
/*                                                                   */
/*  . variables are of the form (n t0 v0 t1 v1 t2 v2) where n is the */
/*    number of assets, ti is the type of an asset, and vi is the    */
/*    money invested in this asset                                   */
/*                                                                   */
/*  . categorical variables are n and the ti's                       */
/*                                                                   */
/*  . with a $10,000 budget, the problem consists in minimizing      */
/*    some measure of the risk and of the revenue                    */
/*                                                                   */
/*  . two classes are defined:                                       */
/*                                                                   */
/*    1. My_Evaluator, wich inherits from the NOMAD class Evaluator, */
/*       in order to define the problem via the virtual function     */
/*       eval_x()                                                    */
/*                                                                   */
/*    2. My_Extended_Poll, which inherits from the NOMAD class       */
/*       Extended_Poll, in order to define the categorical           */
/*       variables neighborhoods, via the virtual function           */
/*       construct_extended_points()                                 */
/*                                                                   */
/*  . My_Extended_Poll also defines 3 signatures, for the solutions  */
/*    with 3, 5, and 7 variables                                     */
/*-------------------------------------------------------------------*/
/*  . compile the scalar version with 'make'                         */
/*  . compile the parallel version with 'make mpi'                   */
/*-------------------------------------------------------------------*/
/*  . execute the scalar version with './categorical'                */
/*  . execute the parallel version with 'mpirun -np p ./categorical' */
/*    with p > 1                                                     */
/*-------------------------------------------------------------------*/
#include "user_functions.h"
#include "lookup_function.h"
#include "My_Evaluator.h"
#include "My_Extended_Poll.h"
#include "nomad.hpp"

#include <sstream>
#include <iterator>
#include <iostream>
#include <string>

#define USE_SURROGATE false


/*------------------------------------------*/
/*            NOMAD main function           */
/*------------------------------------------*/
int main(int argc, char ** argv) {

	// Input arguments
	int n_stages = 6;
	int r_n, eval_point_n;
	double r_thresh_n;

	int call_type, obj_type;
	std::vector<int> r_vec, eval_point;
	std::vector<double> r_thresh;

	call_type = stoi(argv[1]);
	obj_type = stoi(argv[2]);
	std::string weight_file = argv[3];
	std::string res_ip_file = argv[4];
	std::string exc_ip_file = argv[5];
	std::string res_th_file = argv[6];
	std::string exc_th_file = argv[7];
	int n_sargs = 8; // number of static arguments

	for (int r_i = n_sargs; r_i < (n_stages + n_sargs); r_i++) {
		r_n = stoi(argv[r_i]);
		r_vec.push_back(r_n);
	}

	for (int r_i = (n_stages + n_sargs); r_i < (2 * n_stages + n_sargs); r_i++) {
		r_thresh_n = atof(argv[r_i]);
		r_thresh.push_back(r_thresh_n);
	}

	std::string input_file = "./Input_files/" + weight_file;
	std::vector< std::vector<double> > input_data = read_csv_file(input_file); // Make sure there are no empty lines !!
	std::string input_file_th = "./Input_files/" + res_th_file;
	std::vector< std::vector<double> > resiliance_th_data = read_csv_file(input_file_th); // Make sure there are no empty lines !!
	std::string input_file_exc_th = "./Input_files/" + exc_th_file;
	std::vector< std::vector<double> > excess_th_data = read_csv_file(input_file_exc_th); // Make sure there are no empty lines !!

	std::cout << "done importing data" << std::endl;

	// NOMAD initializations:
	NOMAD::begin(argc, argv);

	// display:
	NOMAD::Display out(std::cout);
	out.precision(NOMAD::DISPLAY_PRECISION_STD);

	// check the number of processess:
#ifdef USE_MPI
	if (Slave::get_nb_processes() < 2)
	{
		if (Slave::is_master())
			cerr << "usage: \'mpirun -np p ./categorical\' with p>1"
			<< std::endl;
		end();
		return EXIT_FAILURE;
	}
#endif

	try {

		// parameters creation:
		NOMAD::Parameters p(out);
		//if (call_type == 0) { p.set_DISPLAY_DEGREE(0); } // turn off display
		if (call_type == 1) { p.set_DISPLAY_DEGREE(0); } // turn off display
		if (call_type == 2) { p.set_DISPLAY_DEGREE(0); } // turn off display
		std::vector<NOMAD::bb_output_type> bbot(7);
		bbot[0] = NOMAD::OBJ; // objective
		bbot[1] = NOMAD::PB;  // resiliance constraint
		bbot[2] = NOMAD::PB;  // resiliance constraint
		bbot[3] = NOMAD::PB;  // resiliance constraint
		bbot[4] = NOMAD::PB;  // resiliance constraint
		bbot[5] = NOMAD::PB;  // resiliance constraint
		bbot[6] = NOMAD::PB;  // resiliance constraint

		// initial point
		NOMAD::Point x0(3, 7);
		x0[0] = 1;  // 1 deposit
		x0[1] = 1;  // wave concept
		x0[2] = 1;  // deposit type 1

		p.set_X0(x0);

		NOMAD::Point lb(3);
		NOMAD::Point ub(3);
		// Categorical variables don't need bounds
		//lb[0] = 1; ub[0] = 6;
		lb[1] = 0; ub[1] = 2;
		lb[2] = -1; ub[2] = 4;

		p.set_DIMENSION(3);

		// categorical variables:
		p.set_BB_INPUT_TYPE(0, NOMAD::CATEGORICAL);
		p.set_BB_INPUT_TYPE(1, NOMAD::INTEGER);
		p.set_BB_INPUT_TYPE(2, NOMAD::INTEGER);

		//p.set_DISPLAY_DEGREE ( NOMAD::FULL_DISPLAY );

		p.set_BB_OUTPUT_TYPE(bbot);

		p.set_INITIAL_MESH_SIZE(1.0);

		p.set_LOWER_BOUND(lb);
		p.set_UPPER_BOUND(ub);

		p.set_DISPLAY_STATS("bbe ( sol ) obj");
		p.set_MAX_BB_EVAL(3000);
		p.set_H_MIN(1e-4);
		p.set_DISABLE_MODELS();
		// extended poll trigger:
		p.set_EXTENDED_POLL_ENABLED(true);
		//p.set_EXTENDED_POLL_TRIGGER ( 30 , false );
		p.set_EXTENDED_POLL_TRIGGER(30, false);
		//p.set_VNS_SEARCH(true);
		//p.set_VNS_SEARCH(10);
		// parameters validation:
		p.check();

		// custom evaluator creation:
		My_Evaluator ev(p);

		// Pass parameters to blackbox
		ev.run_type = call_type;
		ev.obj_type = obj_type;
		ev.req_vec = r_vec;
		ev.req_thresh = r_thresh;
		ev.design_data = input_data;
		ev.resiliance_th_data = resiliance_th_data;
		ev.excess_th_data = excess_th_data;

		// extended poll:
		My_Extended_Poll ep(p);

		// clear bb eval log file:
		std::ofstream file("./MADS_output/mads_bb_calls.log", std::ofstream::out);
		file.close();

		// clear opt result file:
		std::ofstream opt_file("./MADS_output/mads_x_opt.log", std::ofstream::out);
		opt_file.close();

		std::cout << "done initializing blackbox" << std::endl;

		// Exceute program type
		if (call_type == 0) { // optimization call 

			// algorithm creation and execution:
			NOMAD::Mads mads(p, &ev, &ep, NULL, NULL);
			mads.run();
			// Write optimal point to file
			const NOMAD::Eval_Point *x_opt;
			x_opt = mads.get_best_feasible();

			// convert eval point to int vector
			std::vector<int> x_opt_vec;
			int coor_i;
			NOMAD::Double coor;

			ofstream output_file("./MADS_output/mads_x_opt.log", ofstream::out);

			if (x_opt) {
				x_opt->display_eval(std::cout);

				for (size_t k = 0; k < x_opt->size(); k++) {
					coor = x_opt->get_coord(int(k));
					coor_i = int(coor.value());
					x_opt_vec.push_back(coor_i);
				}
			}
			else {
				std::cout << "no solution found" << std::endl;
				x_opt_vec.push_back(-1);
				//(output_file) << "-1" << '\n';
			}
			writeTofile(x_opt_vec, &output_file);
			output_file.close();

		}
		else if (call_type == 1) { // evaluation call

			int n_deposits = stoi(argv[2 * n_stages + n_sargs]);
			std::vector<int> eval_point;

			for (int r_i = (2 * n_stages + n_sargs); r_i < (2 * n_stages + n_sargs + n_deposits + 2); r_i++) {
				eval_point_n = stoi(argv[r_i]);
				eval_point.push_back(eval_point_n);
			}

			NOMAD::Eval_Point xt(n_deposits + 2, 7);

			for (int k = 0; k < (eval_point[0] + 2); k++) {
				xt[k] = eval_point[k];
			}

			NOMAD::Double hmax = 20.0;
			bool count = false;

			// Test blackbox
			ev.eval_x(xt, hmax, count);
			ep.construct_extended_points(xt); // debug extended poll
			xt.display_eval(std::cout); // Debug evaluator

			// Output evaluation to file
			ofstream output_file("./MADS_output/eval_point_out.log", ofstream::out);
			output_file.precision(11);
			NOMAD::Point outs;
			outs = xt.get_bb_outputs();
			writeTofile_output(outs, &output_file);
			output_file.close();

		}
		else if (call_type == 2) { // feasiblity call

			std::vector<double> feas_out = feasiblity_loop(input_data, &ev, &ep);
			// Output feasibility check to file
			ofstream feas_file("./MADS_output/feasiblity.log", ofstream::out);
			feas_file.precision(1);
			writeTofile_vector(feas_out, &feas_file);
			feas_file.close();
		}

	}
	catch (exception & e) {
		std::string error = std::string("NOMAD has been interrupted: ") + e.what();
		if (NOMAD::Slave::is_master())
			std::cerr << std::endl << error << std::endl << std::endl;
	}

	NOMAD::Slave::stop_slaves(out);
	NOMAD::end();

	return EXIT_SUCCESS;
}

