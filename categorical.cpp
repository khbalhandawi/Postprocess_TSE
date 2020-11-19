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
//#include "matplotlibcpp.h"
#include <sstream>
#include <iterator>
#include <iostream>
#include <vector>

using namespace std;
//namespace plt = matplotlibcpp;
//using namespace NOMAD;

/*----------------------------------------*/
/*           support functions            */
/*----------------------------------------*/
// This function splits a delimlited string into vector of doubles
vector<double> split_string(string line)
{
	string input = line;
	istringstream ss(input);
	string value;
	double value_db;

	vector<double> out_vector;
	while (getline(ss, value, ','))
	{
		value_db = atof(value.c_str()); // convert to float
		out_vector.push_back(value_db); // Vector of floats
		// cout << token << '\n';
	}

	return out_vector;
}

vector<string> split_string_titles(string line)
{
	string input = line;
	istringstream ss(input);
	string value;

	vector<string> out_vector;
	while (getline(ss, value, ','))
	{
		out_vector.push_back(value.c_str()); // Vector of strings
		// cout << token << '\n';
	}

	return out_vector;
}
//****************End of split_stringe funtion****************

// This function transposes a vector of vectors
vector<vector<double>> transpose(const vector<vector<double> > data) {
	// this assumes that all inner vectors have the same size and
	// allocates space for the complete result in advance
	vector<vector<double>> result(data[0].size(),
		vector<double>(data.size()));
	for (vector<double>::size_type i = 0; i < data[0].size(); i++)
		for (vector<double>::size_type j = 0; j < data.size(); j++) {
			result[i][j] = data[j][i];
		}
	return result;
}
//****************End of transpose vector funtion****************\

// This function reads a csv delimlited file
vector< vector<double> >  read_csv_file(string filename)
{
	ifstream file(filename); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
	string line, value;
	vector<double> line_vec;
	vector<string> title_vec;
	vector< vector<double> > array_grid, array_grid_T;

	int i = 0;
	while (file.good())
	{
		getline(file, line, '\n'); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/


		if (i == 0) {
			title_vec = split_string_titles(line);

			// display titles

			//for (int i = 0; i < title_vec.size(); ++i)
			//	std::cout << title_vec[i] << ' ';
			//std::cout << endl;
		}

		else {

			line_vec = split_string(line);
			size_t n_cols = line_vec.size();

			vector<double> row;
			for (size_t col_i = 0; col_i < n_cols; col_i++) {
				row.push_back(line_vec[col_i]);
			}

			array_grid.push_back(row);

		}

		i++;
	}

	array_grid_T = transpose(array_grid);

	return array_grid_T;
}
//****************End of CSV READ funtion****************

// Function to write permutation vectors to file
bool writeTofile(vector<int> & matrix, ofstream *file)
{
	// Convert int to ostring stream
	ostringstream oss;

	if (!matrix.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		copy(matrix.begin(), matrix.end() - 1,
			ostream_iterator<int>(oss, ","));

		// Now add the last element with no delimiter
		oss << matrix.back();
	}

	// Write ostring stream to file
	if (file->is_open())
	{
		(*file) << oss.str() << '\n';
	}
	else
		return false;
	//file.close();
	return true;
}
//****************End of writeTofile funtion****************

// Function to display permutation vectors to cout
void print_vector(const vector<int> & other_types) {

	ostringstream oss;

	if (!other_types.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		copy(other_types.begin(), other_types.end() - 1,
			ostream_iterator<int>(oss, ","));

		// Now add the last element with no delimiter
		oss << other_types.back();
	}

	cout << oss.str() << endl;

}
//****************End of print_vector funtion****************

#define USE_SURROGATE false

/*----------------------------------------*/
/*               the problem              */
/*----------------------------------------*/
class My_Evaluator : public NOMAD::Evaluator
{
public:
	My_Evaluator(const NOMAD::Parameters & p) :
		Evaluator(p) {}

	~My_Evaluator(void) {}

	bool eval_x(NOMAD::Eval_Point   & x,
		const NOMAD::Double & h_max,
		bool         & count_eval) const;

	int req_index;
	double R_threshold; // reliability threshold
	vector<vector<double>> design_data, resiliance_ip_data, resiliance_th_data;

};


/*--------------------------------------------------*/
/*  user class to define categorical neighborhoods  */
/*--------------------------------------------------*/
class My_Extended_Poll : public NOMAD::Extended_Poll
{

private:

	// signatures for 1, 2, 3 or 4 assets:
	NOMAD::Signature * _s1, *_s2, *_s3, *_s4, *_s5;

public:

	// constructor:
	My_Extended_Poll(NOMAD::Parameters &);

	bool show_ep;

	// destructor:
	virtual ~My_Extended_Poll(void) { delete _s1; delete _s2; delete _s3; delete _s4; }

	// construct the extended poll points:
	virtual void construct_extended_points(const NOMAD::Eval_Point &);

};

/*------------------------------------------*/
/*            NOMAD main function           */
/*------------------------------------------*/
int main(int argc, char ** argv)
{
	
	int n_sargs = 8; // number of static arguments

	// set default arguments if input arguments not provided
	int call_type = 1;
	int req_index = 0;
	double R_threshold = 0.99; // reliability threshold
	string weight_file = "varout_opt_log_R4.log";
	string res_ip_file = "resiliance_ip_R4.log";
	string exc_ip_file = "excess_ip_R4.log";
	string res_th_file = "resiliance_th_R4.log";
	string exc_th_file = "excess_th_R4.log";

	if (argc == n_sargs + 1) {
		int call_type = stoi(argv[1]);
		int req_index = stoi(argv[2]);
		double R_threshold = stod(argv[3]); // reliability threshold
		string weight_file = argv[4];
		string res_ip_file = argv[5];
		string exc_ip_file = argv[6];
		string res_th_file = argv[7];
		string exc_th_file = argv[8];
	}
	else if (argc == 1) {
		call_type = 0;
		n_sargs = 0;
	}
	else
	{
		n_sargs = 0;
	}
	

	string input_file = "./Input_files/" + weight_file;
	vector< vector<double> > input_data = read_csv_file(input_file); // Make sure there are no empty lines !!
	string input_file_ip = "./Input_files/" + res_ip_file;
	vector< vector<double> > resiliance_ip_data = read_csv_file(input_file_ip); // Make sure there are no empty lines !!
	string input_file_exc_ip = "./Input_files/" + exc_ip_file;
	vector< vector<double> > excess_ip_data = read_csv_file(input_file_exc_ip); // Make sure there are no empty lines !!
	string input_file_th = "./Input_files/" + res_th_file;
	vector< vector<double> > resiliance_th_data = read_csv_file(input_file_th); // Make sure there are no empty lines !!
	string input_file_exc_th = "./Input_files/" + exc_th_file;
	vector< vector<double> > excess_th_data = read_csv_file(input_file_exc_th); // Make sure there are no empty lines !!

	// NOMAD initializations:
	NOMAD::begin(argc, argv);

	// display:
	NOMAD::Display out(cout);
	out.precision(NOMAD::DISPLAY_PRECISION_STD);

	// check the number of processess:
#ifdef USE_MPI
	if (Slave::get_nb_processes() < 2)
	{
		if (Slave::is_master())
			cerr << "usage: \'mpirun -np p ./categorical\' with p>1"
			<< endl;
		end();
		return EXIT_FAILURE;
	}
#endif

	try
	{

		// parameters creation:
		NOMAD::Parameters p(out);

		vector<NOMAD::bb_output_type> bbot(2);
		bbot[0] = NOMAD::EB;  // resiliance constraint
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
		NOMAD::Eval_Point xt(3, 2);
		xt[0] = 1;
		xt[1] = 0;
		xt[2] = 1;

		NOMAD::Double hmax = 1;
		bool count = false;

		// custom evaluator creation:
		My_Evaluator ev(p);

		// Pass parameters to blackbox
		ev.req_index = req_index;
		ev.R_threshold = R_threshold;
		ev.design_data = input_data;
		ev.resiliance_ip_data = resiliance_ip_data;
		ev.resiliance_th_data = resiliance_th_data;

		//ev.eval_x(xt, hmax, count);
		//xt.display(cout); // Debug evaluator

		// extended poll:
		My_Extended_Poll ep(p);
		ep.show_ep = false;

		// clear bb eval log file:
		ofstream file("./MADS_output/mads_bb_calls.log", ofstream::out);
		file.close();

		// clear opt result file:
		ofstream opt_file("./MADS_output/mads_x_opt.log", ofstream::out);
		opt_file.close();

		cout << "done initializing blackbox" << endl;

		// Exceute program type
		if (call_type == 0) { // optimization call 

			// algorithm creation and execution:
			NOMAD::Mads mads(p, &ev, &ep, NULL, NULL);
			mads.run();

			// Write optimal point to file
			const NOMAD::Eval_Point *x_opt;
			x_opt = mads.get_best_feasible();

			// convert eval point to int vector
			vector<int> x_opt_vec;
			int coor_i;
			NOMAD::Double coor;

			ofstream output_file("./MADS_output/mads_x_opt.log", ofstream::out);

			if (x_opt) {
				x_opt->display_eval(cout);

				for (size_t k = 0; k < x_opt->size(); k++) {
					coor = x_opt->get_coord(int(k));
					coor_i = int(coor.value());
					x_opt_vec.push_back(coor_i);
				}
			}
			else {
				cout << "no solution found" << endl;
				x_opt_vec.push_back(-1);
				//(output_file) << "-1" << '\n';
			}
			writeTofile(x_opt_vec, &output_file);
			output_file.close();
		}
		else if (call_type == 1) { // evaluation call
			
			ep.show_ep = true; // toggle on show extended poll
			int n_start = n_sargs + 1;
			int n_dim = argc - n_sargs - 1;

			// extract evaluation point values
			int e_n; 
			vector<int> eval_point;

			for (int i = n_start; i < n_start + n_dim; i++) {
				e_n = stoi(argv[i]);
				eval_point.push_back(e_n);
			}

			NOMAD::Eval_Point xt(n_dim + 1, 2);
			// assign variables to NOMAD eval_point object
			xt[0] = n_dim - 1; // set first variable to number of deposits
			for (int k = 1; k < n_dim + 1; k++) {
				xt[k] = eval_point[k - 1];
			}

			NOMAD::Double hmax = 20.0;
			bool count = false;

			// Test blackbox
			ev.eval_x(xt, hmax, count);
			cout << "list of extended poll points" << endl;
			ep.construct_extended_points(xt); // debug extended poll
			cout << "constraint, objective" << endl;
			xt.display_eval(cout); // Debug evaluator

			// Output evaluation to file
			ofstream output_file("./MADS_output/eval_point_out.log", ofstream::out);
			output_file.precision(11);
			NOMAD::Point outs;
			outs = xt.get_bb_outputs();

		}
	}
	catch (exception & e) {
		string error = string("NOMAD has been interrupted: ") + e.what();
		if (NOMAD::Slave::is_master())
			cerr << endl << error << endl << endl;
	}


	NOMAD::Slave::stop_slaves(out);
	NOMAD::end();

	return EXIT_SUCCESS;
}

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
	vector<vector<double>> resiliance_ip_data = My_Evaluator::resiliance_ip_data; // get requirement matrix (IP)
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

/*-----------------------------------------*/
/*  constructor: creates the 4 signatures  */
/*-----------------------------------------*/
My_Extended_Poll::My_Extended_Poll(NOMAD::Parameters & p)
	: Extended_Poll(p),
	_s1(NULL),
	_s2(NULL),
	_s3(NULL),
	_s4(NULL),
	_s5(NULL)
{
	// signature for 1 asset:
	// ----------------------
	vector<NOMAD::bb_input_type> bbit_1(3);
	bbit_1[0] = NOMAD::CATEGORICAL;
	bbit_1[1] = bbit_1[2] = NOMAD::INTEGER;

	const NOMAD::Point & d0_1 = p.get_initial_poll_size();
	const NOMAD::Point & lb_1 = p.get_lb();
	const NOMAD::Point & ub_1 = p.get_ub();

	_s1 = new NOMAD::Signature(3,
		bbit_1,
		d0_1,
		lb_1,
		ub_1,
		p.get_direction_types(),
		p.get_sec_poll_dir_types(),
		p.get_int_poll_dir_types(),
		_p.out());

	// signature for 2 deposits:
	// -----------------------
	{
		vector<NOMAD::bb_input_type> bbit_2(4);
		bbit_2[0] = NOMAD::CATEGORICAL;
		bbit_2[1] = bbit_2[2] = bbit_2[3] = NOMAD::INTEGER;

		NOMAD::Point d0_2(4);
		NOMAD::Point lb_2(4);
		NOMAD::Point ub_2(4);

		// Categorical variables don't need bounds
		for (int i = 0; i < 4; ++i)
		{
			if (i == 0) {
				bbit_2[i] = bbit_1[0];
				d0_2[i] = d0_1[0];
				lb_2[i] = lb_1[0];
				ub_2[i] = ub_1[0];
			}
			else if (i == 1) {
				bbit_2[i] = bbit_1[1];
				d0_2[i] = d0_1[1];
				lb_2[i] = lb_1[1];
				ub_2[i] = ub_1[1];
			}
			else {
				bbit_2[i] = bbit_1[2];
				d0_2[i] = d0_1[2];
				lb_2[i] = lb_1[2];
				ub_2[i] = ub_1[2];
			}
		}

		_s2 = new NOMAD::Signature(4,
			bbit_2,
			d0_2,
			lb_2,
			ub_2,
			p.get_direction_types(),
			p.get_sec_poll_dir_types(),
			p.get_int_poll_dir_types(),
			_p.out());
	}
	// signature for 3 deposits:
	// -----------------------
	{
		vector<NOMAD::bb_input_type> bbit_3(5);
		bbit_3[0] = NOMAD::CATEGORICAL;
		bbit_3[1] = bbit_3[2] = bbit_3[3] = bbit_3[4] = NOMAD::INTEGER;

		NOMAD::Point d0_3(5);
		NOMAD::Point lb_3(5);
		NOMAD::Point ub_3(5);

		// Categorical variables don't need bounds
		for (int i = 0; i < 5; ++i)
		{
			if (i == 0) {
				bbit_3[i] = bbit_1[0];
				d0_3[i] = d0_1[0];
				lb_3[i] = lb_1[0];
				ub_3[i] = ub_1[0];
			}
			else if (i == 1) {
				bbit_3[i] = bbit_1[1];
				d0_3[i] = d0_1[1];
				lb_3[i] = lb_1[1];
				ub_3[i] = ub_1[1];
			}
			else {
				bbit_3[i] = bbit_1[2];
				d0_3[i] = d0_1[2];
				lb_3[i] = lb_1[2];
				ub_3[i] = ub_1[2];
			}
		}

		_s3 = new NOMAD::Signature(5,
			bbit_3,
			d0_3,
			lb_3,
			ub_3,
			p.get_direction_types(),
			p.get_sec_poll_dir_types(),
			p.get_int_poll_dir_types(),
			_p.out());
	}
	// signature for 4 deposits:
	// -----------------------
	{
		vector<NOMAD::bb_input_type> bbit_4(6);
		bbit_4[0] = NOMAD::CATEGORICAL;
		bbit_4[1] = bbit_4[2] = bbit_4[3] = bbit_4[4] = bbit_4[5] = NOMAD::INTEGER;

		NOMAD::Point d0_4(6);
		NOMAD::Point lb_4(6);
		NOMAD::Point ub_4(6);

		// Categorical variables don't need bounds
		for (int i = 0; i < 6; ++i)
		{
			if (i == 0) {
				bbit_4[i] = bbit_1[0];
				d0_4[i] = d0_1[0];
				lb_4[i] = lb_1[0];
				ub_4[i] = ub_1[0];
			}
			else if (i == 1) {
				bbit_4[i] = bbit_1[1];
				d0_4[i] = d0_1[1];
				lb_4[i] = lb_1[1];
				ub_4[i] = ub_1[1];
			}
			else {
				bbit_4[i] = bbit_1[2];
				d0_4[i] = d0_1[2];
				lb_4[i] = lb_1[2];
				ub_4[i] = ub_1[2];
			}
		}

		_s4 = new NOMAD::Signature(6,
			bbit_4,
			d0_4,
			lb_4,
			ub_4,
			p.get_direction_types(),
			p.get_sec_poll_dir_types(),
			p.get_int_poll_dir_types(),
			_p.out());

	}
	// signature for 5 deposits:
	// -----------------------
	{
		vector<NOMAD::bb_input_type> bbit_5(7);
		bbit_5[0] = NOMAD::CATEGORICAL;
		bbit_5[1] = bbit_5[2] = bbit_5[3] = bbit_5[4] = bbit_5[5] = bbit_5[6] = NOMAD::INTEGER;

		NOMAD::Point d0_5(7);
		NOMAD::Point lb_5(7);
		NOMAD::Point ub_5(7);

		// Categorical variables don't need bounds
		for (int i = 0; i < 7; ++i)
		{
			if (i == 0) {
				bbit_5[i] = bbit_1[0];
				d0_5[i] = d0_1[0];
				lb_5[i] = lb_1[0];
				ub_5[i] = ub_1[0];
			}
			else if (i == 1) {
				bbit_5[i] = bbit_1[1];
				d0_5[i] = d0_1[1];
				lb_5[i] = lb_1[1];
				ub_5[i] = ub_1[1];
			}
			else {
				bbit_5[i] = bbit_1[2];
				d0_5[i] = d0_1[2];
				lb_5[i] = lb_1[2];
				ub_5[i] = ub_1[2];
			}
		}

		_s5 = new NOMAD::Signature(7,
			bbit_5,
			d0_5,
			lb_5,
			ub_5,
			p.get_direction_types(),
			p.get_sec_poll_dir_types(),
			p.get_int_poll_dir_types(),
			_p.out());

	}
}

/*--------------------------------------*/
/*  construct the extended poll points  */
/*      (categorical neighborhoods)     */
/*--------------------------------------*/
void My_Extended_Poll::construct_extended_points(const NOMAD::Eval_Point & x) {

	// ep inputs (bool show_ep)
	bool show_ep = My_Extended_Poll::show_ep;
	
	// number of deposits:
	int n = static_cast<int> (x[0].value());
	vector<NOMAD::Point> extended;
	
	// current type of concept:
	int c = static_cast<int> (x[1].value());

	// current type of deposit
	size_t n_var = x.size(); // Accessing last element 
	int cur_type = static_cast<int> (x[n_var - 1].value());

	// list of concepts
	vector<int> concepts = { 0,1,2 };
	vector<int> deposits_c2 = { 0,1,2,3 };
	vector<int> deposits_c1 = { 0,1,2,3,4 };
	vector<int> deposits_c0 = { 0,1,2 };
	vector<int> deposits;
	// this vector contains the types of the other deposits:
	vector<int> other_types, other_concepts, other_types_change, other_types_add;


	// types of deposits available to each concept
	switch (c) {
	case 0:
		deposits = deposits_c0;
		other_concepts.push_back(1);
		other_concepts.push_back(2);
		break;

	case 1:
		deposits = deposits_c1;
		other_concepts.push_back(0);
		other_concepts.push_back(2);
		break;

	case 2:
		deposits = deposits_c2;
		other_concepts.push_back(0);
		other_concepts.push_back(1);
		break;
	}

	// remove existing deposits from available choices:
	for (size_t k = 0; k < (x.size() - 2); ++k) {
		deposits.erase(remove(deposits.begin(), deposits.end(), x[k + 2]), deposits.end());
	}

	other_types = deposits;

	other_types_change = other_types; // do not change into same deposit type
	other_types_change.erase(
		remove(other_types_change.begin(), other_types_change.end(), cur_type),
		other_types_change.end());

	other_types_add = other_types;

	// Extract design vector from decision vector (strip the -1 decisions)
	vector<int> input_deposits; // extract different deposit types
	input_deposits.push_back(int(x[1].value())); // push back concept type

	for (size_t k = 0; k < (x.size() - 2); ++k) {
		input_deposits.push_back(int(x[k + 2].value())); // get input vector
	}

	// remove -1 deposits from input (in MSSP code):
	vector<int> lookup_vector;
	for (size_t m = 0; m < input_deposits.size(); ++m) {
			lookup_vector.push_back(input_deposits[m]);
	}

	bool check_other_concepts = false; // Added other equivalent concepts to extended pool

	// 1 deposit:
	// --------
	if (n == 1) {

		// add 1 deposit (1 or 3 neighbors):
		for (size_t k = 0; k < other_types.size(); ++k) {
			NOMAD::Point y(4);

			y[0] = 2;
			y[1] = c;
			y[2] = cur_type;
			y[3] = other_types[k];

			add_extended_poll_point(y, *_s2);
			extended.push_back(y);
		}

		// change the type of the deposit to the other types (1 or 3 neighbors):
		for (size_t k = 0; k < other_types.size(); ++k)
		{
			NOMAD::Point y = x;
			y[2] = other_types[k];

			add_extended_poll_point(y, *_s1);
			extended.push_back(y);
		}

		
		// loop over concepts
		// Concept 0 is allowed to change to concept 1 or 2
		if (find(deposits_c0.begin(), deposits_c0.end(), cur_type) != deposits_c0.end()) { // change concept allowed if a common deposit is found
			for (size_t j = 0; j < other_concepts.size(); ++j) {

				switch (other_concepts[j]) {
				case 0:
					deposits = deposits_c0;
					other_types = deposits;
					break;
				case 1:
					deposits = deposits_c1;
					other_types = deposits;
					break;
				case 2:
					deposits = deposits_c2;
					other_types = deposits;
					break;
				}

				// change the type of the deposit to the other types (1 or 3 neighbors):
				for (size_t k = 0; k < other_types.size(); ++k) {
					NOMAD::Point y = x;
					y[1] = other_concepts[j];
					y[2] = other_types[k];

					add_extended_poll_point(y, *_s1);
					extended.push_back(y);
				}

			}
		}

		bool exit_loop; // flag to check if invalid concept selected
		// Concept 2 is allowed to change to concept 1 only
		if (find(deposits_c2.begin(), deposits_c2.end(), cur_type) != deposits_c2.end()) { // change concept allowed if a common deposit is found
			for (size_t j = 0; j < other_concepts.size(); ++j) {

				switch (other_concepts[j]) {
				case 1:
					deposits = deposits_c1;
					other_types = deposits;
					break;
				case 2:
					deposits = deposits_c2;
					other_types = deposits;
					break;
				case 0:
					exit_loop = true;
				}

				if (exit_loop) {
					break;
				}
				else {
					// change the type of the deposit to the other types (1 or 3 neighbors):
					for (size_t k = 0; k < other_types.size(); ++k) {
						NOMAD::Point y = x;
						y[1] = other_concepts[j];
						y[2] = other_types[k];

						add_extended_poll_point(y, *_s1);
						extended.push_back(y);
					}
				}

			}
		}

		

		/*
		// check if all deposits are common with another concept
		for (size_t j = 0; j < other_concepts.size(); ++j) {

			switch (other_concepts[j]) {
			case 1:
				deposits = deposits_c1;
				other_types = deposits;
				break;
			case 2:
				deposits = deposits_c2;
				other_types = deposits;
				break;
			case 0:
				deposits = deposits_c0;
				other_types = deposits;
				break;
			}

			bool vector_contained = check_other_concepts;

			for (size_t a = 0; a < lookup_vector.size(); ++a) {// loop over deposit vector

				if (!(find(deposits.begin(), deposits.end(), cur_type) != deposits.end())) {
					vector_contained = false;
				}
			}

			if (vector_contained) {
				// change the concept type:
				NOMAD::Point y = x;
				y[1] = other_concepts[j];

				add_extended_poll_point(y, *_s1);
				extended.push_back(y);
			}

		}

		*/

	}

	// 2 deposits:
	// --------
	else if (n == 2) {

		// remove 1 deposit (1 neighbor):
		{
			NOMAD::Point y(3);
			y[0] = 1;
			y[1] = c;
			y[2] = x[2];

			add_extended_poll_point(y, *_s1);
			extended.push_back(y);
		}

		// change the type of one deposit (2 neighbors):
		for (size_t k = 0; k < other_types_change.size(); ++k) {
			NOMAD::Point y = x;
			y[3] = other_types_change[k];

			add_extended_poll_point(y, *_s2);
			extended.push_back(y);
		}

		// add one deposit (2 neighbors):
		for (size_t k = 0; k < other_types.size(); ++k) {
			NOMAD::Point y(5);
			y[0] = 3;
			y[1] = c;
			y[2] = x[2];
			y[3] = cur_type;
			y[4] = other_types[k];

			add_extended_poll_point(y, *_s3);
			extended.push_back(y);
		}


		// check if all deposits are common with another concept
		for (size_t j = 0; j < other_concepts.size(); ++j) {

			switch (other_concepts[j]) {
			case 1:
				deposits = deposits_c1;
				other_types = deposits;
				break;
			case 2:
				deposits = deposits_c2;
				other_types = deposits;
				break;
			case 0:
				deposits = deposits_c0;
				other_types = deposits;
				break;
			}

			bool vector_contained = check_other_concepts;

			for (size_t a = 0; a < lookup_vector.size(); ++a) {// loop over deposit vector

				if (!(find(deposits.begin(), deposits.end(), cur_type) != deposits.end())) {
					vector_contained = false;
				}
			}

			if (vector_contained) {
				// change the concept type:
				NOMAD::Point y = x;
				y[1] = other_concepts[j];

				add_extended_poll_point(y, *_s2);
				extended.push_back(y);
			}

		}

	}

	// 3 deposits:
	// ---------
	else if (n == 3)
	{

		// remove 1 deposit (1 neighbor):
		{
			NOMAD::Point y(4);
			y[0] = 2;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];

			add_extended_poll_point(y, *_s2);
			extended.push_back(y);
		}

		// change the type of one deposit (1 neighbor):
		for (size_t k = 0; k < other_types.size(); ++k)
		{
			NOMAD::Point y = x;
			y[4] = other_types[k];

			add_extended_poll_point(y, *_s3);
			extended.push_back(y);
		}

		// add one deposit (1 neighbor):
		for (size_t k = 0; k < other_types.size(); ++k)
		{
			NOMAD::Point y(6);
			y[0] = 4;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];
			y[4] = cur_type;
			y[5] = other_types[k];

			add_extended_poll_point(y, *_s4);
			extended.push_back(y);
		}

		// check if all deposits are common with another concept
		for (size_t j = 0; j < other_concepts.size(); ++j) {

			switch (other_concepts[j]) {
			case 1:
				deposits = deposits_c1;
				other_types = deposits;
				break;
			case 2:
				deposits = deposits_c2;
				other_types = deposits;
				break;
			case 0:
				deposits = deposits_c0;
				other_types = deposits;
				break;
			}

			bool vector_contained = check_other_concepts;

			for (size_t a = 0; a < lookup_vector.size(); ++a) {// loop over deposit vector

				if (!(find(deposits.begin(), deposits.end(), cur_type) != deposits.end())) {
					vector_contained = false;
				}
			}

			if (vector_contained) {
				// change the concept type:
				NOMAD::Point y = x;
				y[1] = other_concepts[j];

				add_extended_poll_point(y, *_s3);
				extended.push_back(y);
			}

		}

	}

	// 4 deposits:
	// ---------
	else if (n == 4)
	{

		// remove 1 deposit (1 neighbor):
		{
			NOMAD::Point y(5);
			y[0] = 3;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];
			y[4] = x[4];

			add_extended_poll_point(y, *_s3);
			extended.push_back(y);
		}

		// change the type of one deposit (1 neighbor):
		for (size_t k = 0; k < other_types.size(); ++k)
		{
			NOMAD::Point y = x;
			y[5] = other_types[k];

			add_extended_poll_point(y, *_s4);
			extended.push_back(y);
		}

		// add one deposit (1 neighbor):
		for (size_t k = 0; k < other_types.size(); ++k)
		{
			NOMAD::Point y(7);
			y[0] = 5;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];
			y[4] = x[4];
			y[5] = cur_type;
			y[6] = other_types[k];

			add_extended_poll_point(y, *_s5);
			extended.push_back(y);
		}

		// check if all deposits are common with another concept
		for (size_t j = 0; j < other_concepts.size(); ++j) {

			switch (other_concepts[j]) {
			case 1:
				deposits = deposits_c1;
				other_types = deposits;
				break;
			case 2:
				deposits = deposits_c2;
				other_types = deposits;
				break;
			case 0:
				deposits = deposits_c0;
				other_types = deposits;
				break;
			}

			bool vector_contained = check_other_concepts;

			for (size_t a = 0; a < lookup_vector.size(); ++a) {// loop over deposit vector

				if (!(find(deposits.begin(), deposits.end(), cur_type) != deposits.end())) {
					vector_contained = false;
				}
			}

			if (vector_contained) {
				// change the concept type:
				NOMAD::Point y = x;
				y[1] = other_concepts[j];

				add_extended_poll_point(y, *_s4);
				extended.push_back(y);
			}

		}

	}

	// 5 deposits:
	// ---------
	else if (n == 5) {

		// remove one deposit (1 neighbor):
		NOMAD::Point y(6);
		y[0] = 4;
		y[1] = c;
		y[2] = x[2];
		y[3] = x[3];
		y[4] = x[4];
		y[5] = x[5];

		add_extended_poll_point(y, *_s4);
		extended.push_back(y);

		// check if all deposits are common with another concept
		for (size_t j = 0; j < other_concepts.size(); ++j) {

			switch (other_concepts[j]) {
			case 1:
				deposits = deposits_c1;
				other_types = deposits;
				break;
			case 2:
				deposits = deposits_c2;
				other_types = deposits;
				break;
			case 0:
				deposits = deposits_c0;
				other_types = deposits;
				break;
			}

			bool vector_contained = check_other_concepts;

			for (size_t a = 0; a < lookup_vector.size(); ++a) {// loop over deposit vector

				if (!(find(deposits.begin(), deposits.end(), cur_type) != deposits.end())) {
					vector_contained = false;
				}
			}

			if (vector_contained) {
				// change the concept type:
				NOMAD::Point y = x;
				y[1] = other_concepts[j];

				add_extended_poll_point(y, *_s5);
				extended.push_back(y);
			}

		}

	}

	// display extended poll points
	if (show_ep) {
		for (size_t k = 0; k < extended.size(); k++) {

			NOMAD::Point p = extended[k];
			cout << p << endl;

		}
	}

}
