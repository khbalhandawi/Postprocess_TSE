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
#include "nomad.hpp"
#include "user_functions.h"
#include "My_Evaluator_biobj.h"
#include "My_Extended_Poll_biobj.h"

#include <sstream>
#include <iterator>
#include <iostream>
#include <vector>

#define USE_SURROGATE false

/*------------------------------------------*/
/*            NOMAD main function           */
/*------------------------------------------*/
int main(int argc, char ** argv)
{

	int req_index = stoi(argv[1]);

	std::string weight_file = argv[2];
	std::string res_ip_file = argv[3];
	std::string exc_ip_file = argv[4];
	std::string res_th_file = argv[5];
	std::string exc_th_file = argv[6];

	std::string input_file = "./Input_files/" + weight_file;
	std::vector< std::vector<double> > input_data = read_csv_file(input_file); // Make sure there are no empty lines !!(input_file_exc_ip); // Make sure there are no empty lines !!
	std::string input_file_th = "./Input_files/" + res_th_file;
	std::vector< std::vector<double> > resiliance_th_data = read_csv_file(input_file_th); // Make sure there are no empty lines !!
	std::string input_file_exc_th = "./Input_files/" + exc_th_file;
	std::vector< std::vector<double> > excess_th_data = read_csv_file(input_file_exc_th); // Make sure there are no empty lines !!

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

	try
	{

		// parameters creation:
		NOMAD::Parameters p(out);

		std::vector<NOMAD::bb_output_type> bbot(2);
		bbot[0] = NOMAD::OBJ; // objective
		bbot[1] = NOMAD::OBJ; // objective

		// initial point
		NOMAD::Point x0(3, 1);
		x0[0] = 1;   // 1 deposit
		x0[1] = 1;   // wave concept
		x0[2] = 1;   // deposit type 1
		p.set_X0(x0);

		NOMAD::Point lb(3);
		NOMAD::Point ub(3);
		// Categorical variables don't need bounds
		lb[1] = 0; ub[1] = 2;
		lb[2] = 0; ub[2] = 4;

		//p.set_DISPLAY_DEGREE ( NOMAD::FULL_DISPLAY );

		p.set_DIMENSION(3);

		p.set_BB_OUTPUT_TYPE(bbot);

		// categorical variables:
		p.set_BB_INPUT_TYPE(0, NOMAD::CATEGORICAL);
		p.set_BB_INPUT_TYPE(1, NOMAD::INTEGER);
		p.set_BB_INPUT_TYPE(2, NOMAD::INTEGER);

		p.set_DISABLE_MODELS();
		p.set_INITIAL_MESH_SIZE(2.0);

		p.set_LOWER_BOUND(lb);
		p.set_UPPER_BOUND(ub);

		p.set_DISPLAY_STATS("bbe ( sol ) obj");
		p.set_MAX_BB_EVAL(500);
		p.set_MULTI_NB_MADS_RUNS(20);

		// extended poll trigger:
		p.set_EXTENDED_POLL_TRIGGER(1, false);

		// parameters validation:
		p.check();

		// custom point to debug ev and ep:
		NOMAD::Eval_Point xt(3, 1);
		xt[0] = 1;
		xt[1] = 0;
		xt[2] = 1;

		NOMAD::Double hmax = 1;
		bool count = false;

		// custom evaluator creation:
		My_Evaluator ev(p);

		// Pass parameters to blackbox
		ev.req_index = req_index;
		ev.design_data = input_data;
		ev.resiliance_th_data = resiliance_th_data;

		//ev.eval_x(xt, hmax, count);
		//xt.display(std::cout); // Debug evaluator

		// extended poll:
		My_Extended_Poll ep(p);

		ep.construct_extended_points(xt); // debug extended poll

		// clear bb eval log file:
		ofstream file("./MADS_output/mads_bb_calls_biobj.log", ofstream::out);
		file.close();

		// algorithm creation and execution:
		NOMAD::Mads mads(p, &ev, &ep, NULL, NULL);
		mads.multi_run();

		// Write optimal pareto points to file
		const NOMAD::Pareto_Front *x_pareto;

		x_pareto = mads.get_pareto_front();

		ofstream output_file("./MADS_output/mads_x_opt_pareto.log", ofstream::out);
		output_file.app;

		int n_pareto = x_pareto->size();
		x_pareto->begin()->display(output_file);
		output_file << "\n";

		for (int i = 1; i < n_pareto; ++i) {
			x_pareto->next()->display(output_file);
			output_file << "\n";
		}

		output_file.close();

	}
	catch (exception & e) {
		std::string error = std::string("NOMAD has been interrupted: ") + e.what();
		if (NOMAD::Slave::is_master())
			cerr << std::endl << error << std::endl << std::endl;
	}


	NOMAD::Slave::stop_slaves(out);
	NOMAD::end();

	return EXIT_SUCCESS;
}