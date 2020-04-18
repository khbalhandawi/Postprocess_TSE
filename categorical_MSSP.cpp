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
vector<double> split_string(string line) {
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

vector<string> split_string_titles(string line) {
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
vector< vector<double> >  read_csv_file(string filename) {
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

// Function to display double vectors to cout
void print_vec_double(std::vector<double> const &input) {
	for (int i = 0; i < input.size(); i++) {
		cout << input.at(i) << ' ';
	}
	cout << endl;
}
//****************End of print_vec_double funtion****************

// Function to write permutation vectors to file
bool writeTofile(vector<int> & matrix, ofstream *file) {

	int n_vars = 8;
	vector<int> matrix_us; // unstripped matrix

	for (size_t k = 0; k < (n_vars); ++k) {

		if (k < matrix.size()) {
			matrix_us.push_back(matrix[k]);
		}
		else {
			matrix_us.push_back(-1); // append -1 to remaining entries
		}
	}

	// Convert int to ostring stream
	ostringstream oss;

	if (!matrix_us.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		copy(matrix_us.begin(), matrix_us.end() - 1,
			ostream_iterator<int>(oss, ","));

		// Now add the last element with no delimiter
		oss << matrix_us.back();
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

// Function to write output vectors to file
bool writeTofile_output(NOMAD::Point & matrix, ofstream *file) {

	size_t n_outs = matrix.size();
	vector<double> matrix_out; // unstripped matrix

	for (size_t k = 0; k < (n_outs); ++k)
	{
		matrix_out.push_back(matrix[k].value());
	}

	// Convert int to ostring stream
	ostringstream oss;
	oss.precision(11);
	if (!matrix_out.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		copy(matrix_out.begin(), matrix_out.end() - 1,
			ostream_iterator<double>(oss, ","));

		// Now add the last element with no delimiter
		oss << matrix_out.back();
	}
	file->precision(11);
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

// Function to write output vectors to file
bool writeTofile_vector(vector<double> & matrix, ofstream *file) {

	// Convert int to ostring stream
	ostringstream oss;
	if (!matrix.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		copy(matrix.begin(), matrix.end() - 1,
			ostream_iterator<double>(oss, ","));

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
vector<double> lookup_function(const vector<int> & input_deposits,
							   const vector<vector<double>> & design_data,
							   const vector<double> & resiliance_th) {

	double W, R, count_eval;

	vector<double> concept, i1, i2, i3, i4, n_f_th, weight;

	concept = design_data[1];
	i1 = design_data[2];
	i2 = design_data[3];
	i3 = design_data[4];
	i4 = design_data[5];
	n_f_th = design_data[32];
	weight = design_data[34];
	//resiliance_th = design_data[45];
	// number of deposits:
	int value;
	
	vector<int> input;
	for (int i = 0; i < 5; ++i)
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
	vector<int> lookup;

	bool found = false;
	for (int i = 0; i < k; ++i)
	{
		lookup = { static_cast<int> (concept[i]), static_cast<int> (i1[i]),
			static_cast<int> (i2[i]), static_cast<int> (i3[i]),
			static_cast<int> (i4[i]) };

		if (input == lookup) {
			W = weight[i];
			R = resiliance_th[i];
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

	return { W, R, count_eval };

}
//****************End of lookup_function funtion****************

#define USE_SURROGATE false

/*----------------------------------------*/
/*               the problem              */
/*----------------------------------------*/
class My_Evaluator : public NOMAD::Evaluator {
public:
	My_Evaluator ( const NOMAD::Parameters & p ) :
    Evaluator ( p ) {}
	
	~My_Evaluator ( void ) {}

	bool eval_x (NOMAD::Eval_Point   & x          ,
				 const NOMAD::Double & h_max      ,
				 bool         & count_eval        ) const;

	int run_type;
	vector<int> req_vec;
	vector<double> req_thresh;
	vector<vector<double>> design_data, resiliance_ip_data, resiliance_th_data;

};


/*--------------------------------------------------*/
/*  user class to define categorical neighborhoods  */
/*--------------------------------------------------*/
class My_Extended_Poll : public NOMAD::Extended_Poll {
	
	private:
	
	// signatures for 1, 2, 3 or 4 assets:
	NOMAD::Signature * _s1, * _s2, * _s3, * _s4, *_s5, *_s6;

	public:
	
	// constructor:
	My_Extended_Poll (NOMAD::Parameters & );
	
	// destructor:
	virtual ~My_Extended_Poll ( void ) { delete _s1; delete _s2; delete _s3; delete _s4; delete _s5; delete _s6; }
	
	// construct the extended poll points:
	virtual void construct_extended_points ( const NOMAD::Eval_Point & );
	
	// Generate additional padded neighbourhoods:
	virtual void shuffle_padding( const NOMAD::Eval_Point & x, vector<NOMAD::Point> *extended );
	virtual void shuffle_padding(const NOMAD::Point & x, vector<NOMAD::Point> *extended);

	// Generate filled neighbourhoods:
	virtual void fill_point(const NOMAD::Eval_Point & x, vector<NOMAD::Point> *extended);
	virtual void fill_point(const NOMAD::Point & x, vector<NOMAD::Point> *extended);

};


// Function to display check feasiblity of each design
vector<double> feasiblity_loop(const vector<vector<double>> & design_data,
	My_Evaluator *ev,
	My_Extended_Poll *ep) {

	vector<double> concept, i1, i2, i3, i4, n_f_th, weight;

	concept = design_data[1];
	i1 = design_data[2];
	i2 = design_data[3];
	i3 = design_data[4];
	i4 = design_data[5];
	n_f_th = design_data[32];
	weight = design_data[34];

	// number of branches
	size_t k = n_f_th.size();

	// Look up safety factor value
	vector<int> lookup;
	vector<double> feasiblity_vec;

	// loop over designs to check their feasiblity
	for (int i = 0; i < k; ++i) {

		lookup = { static_cast<int> (concept[i]), static_cast<int> (i1[i]),
			static_cast<int> (i2[i]), static_cast<int> (i3[i]),
			static_cast<int> (i4[i]) };

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
		vector<NOMAD::Point> extended;
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
//****************End of feasiblity_loop funtion****************

/*------------------------------------------*/
/*            NOMAD main function           */
/*------------------------------------------*/
int main ( int argc , char ** argv ) {

	// Input arguments
	int n_stages = 6;
	int r_n, eval_point_n;
	double r_thresh_n;

	int call_type;
	vector<int> r_vec, eval_point;
	vector<double> r_thresh;
	
	call_type = stoi(argv[1]);
	int n_sargs = 2; // number of static arguments

	for (int r_i = n_sargs; r_i < (n_stages + n_sargs); r_i++) {
		r_n = stoi(argv[r_i]);
		r_vec.push_back(r_n);
	}

	for (int r_i = (n_stages + n_sargs); r_i < (2 * n_stages + n_sargs); r_i++) {
		r_thresh_n = atof(argv[r_i]);
		r_thresh.push_back(r_thresh_n);
	}

	string input_file = "./Input_files/varout_opt_log.log";
	vector< vector<double> > input_data = read_csv_file(input_file); // Make sure there are no empty lines !!
	string input_file_ip = "./Input_files/resiliance_ip.log";
	vector< vector<double> > resiliance_ip_data = read_csv_file(input_file_ip); // Make sure there are no empty lines !!
	string input_file_th = "./Input_files/resiliance_th.log";
	vector< vector<double> > resiliance_th_data = read_csv_file(input_file_th); // Make sure there are no empty lines !!

	// NOMAD initializations:
	NOMAD::begin (argc, argv);
	
	// display:
	NOMAD::Display out(cout);
	out.precision(NOMAD::DISPLAY_PRECISION_STD);

	// check the number of processess:
	#ifdef USE_MPI
		if ( Slave::get_nb_processes() < 2 )
		{
			if ( Slave::is_master() )
				cerr << "usage: \'mpirun -np p ./categorical\' with p>1"
				<< endl;
			end();
			return EXIT_FAILURE;
		}
	#endif
	
	try {
		
		// parameters creation:
		NOMAD::Parameters p ( out );
		//if (call_type == 2) { p.set_DISPLAY_DEGREE(0); } // turn off display
		vector<NOMAD::bb_output_type> bbot(7);
		bbot[0] = NOMAD::OBJ; // objective
		bbot[1] = NOMAD::PB;  // resiliance constraint
		bbot[2] = NOMAD::PB;  // resiliance constraint
		bbot[3] = NOMAD::PB;  // resiliance constraint
		bbot[4] = NOMAD::PB;  // resiliance constraint
		bbot[5] = NOMAD::PB;  // resiliance constraint
		bbot[6] = NOMAD::PB;  // resiliance constraint

		// initial point
		NOMAD::Point x0(3, 7);
		x0[0] =  1;  // 1 deposit
		x0[1] =  0;  // wave concept
		x0[2] =  1;  // deposit type 1

		p.set_X0(x0);

		NOMAD::Point lb(3);
		NOMAD::Point ub(3);
		// Categorical variables don't need bounds
		//lb[0] = 1; ub[0] = 6;
		lb[1] = 0; ub[1] = 1;
		lb[2] = -1; ub[2] = 3;

		p.set_DIMENSION(3);

		// categorical variables:
		p.set_BB_INPUT_TYPE(0, NOMAD::CATEGORICAL);
		p.set_BB_INPUT_TYPE(1, NOMAD::INTEGER);
		p.set_BB_INPUT_TYPE(2, NOMAD::INTEGER);

        //p.set_DISPLAY_DEGREE ( NOMAD::FULL_DISPLAY );

		p.set_BB_OUTPUT_TYPE ( bbot );

		p.set_INITIAL_MESH_SIZE(2.0);

		p.set_LOWER_BOUND ( lb );
		p.set_UPPER_BOUND ( ub );
		
		p.set_DISPLAY_STATS ( "bbe ( sol ) obj" );
		p.set_MAX_BB_EVAL ( 1200 );
		p.set_H_MIN ( 1e-4 );
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
		My_Evaluator ev (p);

		// Pass parameters to blackbox
		ev.run_type = call_type;
		ev.req_vec = r_vec;
		ev.req_thresh = r_thresh;
		ev.design_data = input_data;
		ev.resiliance_ip_data = resiliance_ip_data;
		ev.resiliance_th_data = resiliance_th_data;

		// extended poll:
		My_Extended_Poll ep ( p );

		// clear bb eval log file:
		ofstream file("./MADS_output/mads_bb_calls.log", ofstream::out);
		file.close();

		// clear opt result file:
		ofstream opt_file("./MADS_output/mads_x_opt.log", ofstream::out);
		opt_file.close();

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

			int n_deposits = stoi(argv[2 * n_stages + n_sargs]);
			vector<int> eval_point;

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
			xt.display_eval(cout); // Debug evaluator

			// Output evaluation to file
			ofstream output_file("./MADS_output/eval_point_out.log", ofstream::out);
			output_file.precision(11);
			NOMAD::Point outs;
			outs = xt.get_bb_outputs();
			writeTofile_output(outs, &output_file);
			output_file.close();

		}
		else if (call_type == 2) { // feasiblity call

			vector<double> feas_out = feasiblity_loop(input_data, &ev, &ep);
			// Output feasibility check to file
			ofstream feas_file("./MADS_output/feasiblity.log", ofstream::out);
			feas_file.precision(1);
			writeTofile_vector(feas_out, &feas_file);
			feas_file.close();
		}

	} catch ( exception & e ) {
		string error = string ( "NOMAD has been interrupted: " ) + e.what();
		if (NOMAD::Slave::is_master() )
			cerr << endl << error << endl << endl;
	}
	
	NOMAD::Slave::stop_slaves ( out );
	NOMAD::end();
	
	return EXIT_SUCCESS;
}

/*----------------------------------------------------*/
/*                         eval_x                     */
/*----------------------------------------------------*/
bool My_Evaluator::eval_x(NOMAD::Eval_Point   & x,
						  const NOMAD::Double & h_max,
						  bool         & count_eval) const {

	NOMAD::Double f, g1; // objective function

	int run_type = My_Evaluator::run_type;
	vector<int> r_vec = My_Evaluator::req_vec; // get requirement vector
	vector<double> r_thresh = My_Evaluator::req_thresh; // get threshold vector
	vector<vector<double>> design_data = My_Evaluator::design_data; // get design data
	vector<vector<double>> resiliance_ip_data = My_Evaluator::resiliance_ip_data; // get requirement matrix (IP)
	vector<vector<double>> resiliance_th_data = My_Evaluator::resiliance_th_data; // get requirement matrix (TH)
	//--------------------------------------------------------------//
	// Read from data file

	count_eval = false;
	bool count_eval_lookup;
	vector<double> outputs, W_vector, R_vector;

	int n_stages = 6;
	double W_vector_sum = 0.0;
	vector<int> input_deposits; // extract different deposit types
	input_deposits.push_back(int(x[1].value())); // push back concept type
	int req_index;
	vector<double> resiliance_th;

	if (x[0] == (x.size() - 2)) {
		for (size_t k = 0; k < (n_stages); ++k) {

			req_index = req_vec[k];
			resiliance_th = resiliance_th_data[req_index + 5];

			if (k < (x.size() - 2)) {
				input_deposits.push_back(int(x[k + 2].value())); // get input vector
			}

			// remove -1 deposits from input:
			vector<int> lookup_vector;
			for (size_t m = 0; m < input_deposits.size(); ++m) {
				if (input_deposits[m] != -1) {
					lookup_vector.push_back(input_deposits[m]);
				}
			}

			outputs = lookup_function(lookup_vector, design_data, resiliance_th);

			if (outputs[2] == 1.0) {
				W_vector.push_back(outputs[0]);
				R_vector.push_back(outputs[1]);
				count_eval_lookup = true;
				count_eval = true;

				//cout << outputs[0] << endl;
				//cout << outputs[1] << endl;

			}
			else if (outputs[2] == 0.0) {
				count_eval_lookup = false;
				count_eval = true;
				break;
			}


			W_vector_sum += W_vector.back();
		}
	}

	//vector<double> Resiliance_constraint = { 0.4,0.1,0.1,0.1,0.1,0.99 };
	vector<double> Resiliance_constraint = r_thresh;

	if (count_eval_lookup) {
		x.set_bb_output(0, W_vector_sum);
		//x.set_bb_output(0, W_vector[5]);

		// assign constraint vector:
		vector<int> g_vector;
		NOMAD::Double g_n;
		for (size_t n = 0; n < Resiliance_constraint.size(); ++n) {
			g_n = Resiliance_constraint[n] - R_vector[n];
			x.set_bb_output(int(n + 1), g_n);
		}
		vector<int> output_log; // map input variable to output vector
		for (size_t k = 0; k < x.size(); ++k) {
			output_log.push_back(int(x[k].value())); // populate output vector
		}

		ofstream file("./MADS_output/mads_bb_calls.log", ofstream::app);
		writeTofile(output_log, &file);

	} else if (!count_eval_lookup) {
		x.set_bb_output(0, NOMAD::INF);
		x.set_bb_output(1, NOMAD::INF);
		x.set_bb_output(2, NOMAD::INF);
		x.set_bb_output(3, NOMAD::INF);
		x.set_bb_output(4, NOMAD::INF);
		x.set_bb_output(5, NOMAD::INF);
		x.set_bb_output(6, NOMAD::INF);

		vector<double> W_vector(n_stages, 0.0);

	}

	if (run_type == 1) 	{ // if run type is an evaluation
		ofstream weight_file("./MADS_output/weight_design.log", ofstream::out);
		writeTofile_vector(W_vector, &weight_file);
		weight_file.close();
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
	_s5(NULL),
	_s6(NULL) {

	NOMAD::bb_input_type n_deposits_sig = NOMAD::CATEGORICAL;
	NOMAD::bb_input_type concept_sig = NOMAD::INTEGER;
	NOMAD::bb_input_type deposit_sig = NOMAD::INTEGER;

	// get signature for initial point
	const NOMAD::Point & d0_0 = p.get_initial_poll_size();
	const NOMAD::Point & lb_0 = p.get_lb();
	const NOMAD::Point & ub_0 = p.get_ub();

	// signature for 1 stage:
	// ----------------------
	vector<NOMAD::bb_input_type> bbit_1(3);
	bbit_1[0] = n_deposits_sig;
	bbit_1[1] = concept_sig;
	bbit_1[2] = deposit_sig;

	NOMAD::Point d0_1(3);
	NOMAD::Point lb_1(3);
	NOMAD::Point ub_1(3);

	for (int i = 0; i < 3; ++i) {
		if (i == 0) {
			d0_1[i] = d0_0[0];
			lb_1[i] = lb_0[0];
			ub_1[i] = ub_0[0];
		}
		else if (i == 1) {
			d0_1[i] = d0_0[1];
			lb_1[i] = lb_0[1];
			ub_1[i] = ub_0[1];
		}
		else {
			d0_1[i] = d0_0[2];
			lb_1[i] = lb_0[2];
			ub_1[i] = ub_0[2];
		}
	}

	_s1 = new NOMAD::Signature(3,
		bbit_1,
		d0_1,
		lb_1,
		ub_1,
		p.get_direction_types(),
		p.get_sec_poll_dir_types(),
		p.get_int_poll_dir_types(),
		_p.out());

	// signature for 2 stages:
	// -----------------------
	{
		vector<NOMAD::bb_input_type> bbit_2(4);
		bbit_2[0] = n_deposits_sig;
		bbit_2[1] = concept_sig;
		bbit_2[2] = bbit_2[3] = deposit_sig;

		NOMAD::Point d0_2(4);
		NOMAD::Point lb_2(4);
		NOMAD::Point ub_2(4);

		// Categorical variables don't need bounds
		for (int i = 0; i < 4; ++i) {
			if (i == 0) {
				d0_2[i] = d0_0[0];
				lb_2[i] = lb_0[0];
				ub_2[i] = ub_0[0];
			}
			else if (i == 1) {
				d0_2[i] = d0_0[1];
				lb_2[i] = lb_0[1];
				ub_2[i] = ub_0[1];
			}
			else {
				d0_2[i] = d0_0[2];
				lb_2[i] = lb_0[2];
				ub_2[i] = ub_0[2];
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
	// signature for 3 stages:
	// -----------------------
	{
		vector<NOMAD::bb_input_type> bbit_3(5);
		bbit_3[0] = n_deposits_sig;
		bbit_3[1] = concept_sig;
		bbit_3[2] = bbit_3[3] = bbit_3[4] = deposit_sig;

		NOMAD::Point d0_3(5);
		NOMAD::Point lb_3(5);
		NOMAD::Point ub_3(5);

		// Categorical variables don't need bounds
		for (int i = 0; i < 5; ++i) {
			if (i == 0) {
				d0_3[i] = d0_0[0];
				lb_3[i] = lb_0[0];
				ub_3[i] = ub_0[0];
			}
			else if (i == 1) {
				d0_3[i] = d0_0[1];
				lb_3[i] = lb_0[1];
				ub_3[i] = ub_0[1];
			}
			else {
				d0_3[i] = d0_0[2];
				lb_3[i] = lb_0[2];
				ub_3[i] = ub_0[2];
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
	// signature for 4 stages:
	// -----------------------
	{
		vector<NOMAD::bb_input_type> bbit_4(6);
		bbit_4[0] = n_deposits_sig;
		bbit_4[1] = concept_sig;
		bbit_4[2] = bbit_4[3] = bbit_4[4] = bbit_4[5] = deposit_sig;

		NOMAD::Point d0_4(6);
		NOMAD::Point lb_4(6);
		NOMAD::Point ub_4(6);

		// Categorical variables don't need bounds
		for (int i = 0; i < 6; ++i) {
			if (i == 0) {
				d0_4[i] = d0_0[0];
				lb_4[i] = lb_0[0];
				ub_4[i] = ub_0[0];
			}
			else if (i == 1) {
				d0_4[i] = d0_0[1];
				lb_4[i] = lb_0[1];
				ub_4[i] = ub_0[1];
			}
			else {
				d0_4[i] = d0_0[2];
				lb_4[i] = lb_0[2];
				ub_4[i] = ub_0[2];
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

	// signature for 5 stages:
	// -----------------------
	{
		vector<NOMAD::bb_input_type> bbit_5(7);
		bbit_5[0] = n_deposits_sig;
		bbit_5[1] = concept_sig;
		bbit_5[2] = bbit_5[3] = bbit_5[4] = bbit_5[5] = bbit_5[6] = deposit_sig;

		NOMAD::Point d0_5(7);
		NOMAD::Point lb_5(7);
		NOMAD::Point ub_5(7);

		// Categorical variables don't need bounds
		for (int i = 0; i < 7; ++i) {
			if (i == 0) {
				d0_5[i] = d0_0[0];
				lb_5[i] = lb_0[0];
				ub_5[i] = ub_0[0];
			}
			else if (i == 1) {
				d0_5[i] = d0_0[1];
				lb_5[i] = lb_0[1];
				ub_5[i] = ub_0[1];
			}
			else {
				d0_5[i] = d0_0[2];
				lb_5[i] = lb_0[2];
				ub_5[i] = ub_0[2];
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

	// signature for 6 stages:
	// -----------------------
	{
		vector<NOMAD::bb_input_type> bbit_6(8);
		bbit_6[0] = n_deposits_sig;
		bbit_6[1] = concept_sig;
		bbit_6[2] = bbit_6[3] = bbit_6[4] = bbit_6[5] = bbit_6[6] = bbit_6[7] = deposit_sig;

		NOMAD::Point d0_6(8);
		NOMAD::Point lb_6(8);
		NOMAD::Point ub_6(8);

		// Categorical variables don't need bounds
		for (int i = 0; i < 8; ++i) {
			if (i == 0) {
				d0_6[i] = d0_0[0];
				lb_6[i] = lb_0[0];
				ub_6[i] = ub_0[0];
			}
			else if (i == 1) {
				d0_6[i] = d0_0[1];
				lb_6[i] = lb_0[1];
				ub_6[i] = ub_0[1];
			}
			else {
				d0_6[i] = d0_0[2];
				lb_6[i] = lb_0[2];
				ub_6[i] = ub_0[2];
			}
		}

		_s6 = new NOMAD::Signature(8,
			bbit_6,
			d0_6,
			lb_6,
			ub_6,
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

void insert_i(vector<int> in_list, int n_insert, int n_deposit, vector< vector<int> > & outputs, int depth = 0, int shift = 0) {
		
	depth = depth + 1;
	vector<int> temp, out;
	int i = -1;
	for (int k = 1 + shift; k < n_insert + n_deposit; k++) {
		temp = in_list;
		if (k > temp.size()) {
			temp.insert(temp.end(), i);
		}
		else {
			temp.insert(temp.begin()+k, i);
		}

		out = temp;
		if (depth == n_insert) {
			// stopping condition
			outputs.push_back(out);
		}
		else {
			// recurisive call to go into a nest loop
			insert_i(out, n_insert, n_deposit, outputs, depth, k);
		}
		
	}
}

void fill_i(vector<int> in_list, int n_stages, vector< vector<int> > & outputs) {

	vector<int> temp, out;
	int i = -1;
	temp = in_list;

	for (int k = n_stages; k < 6; k++) {
		temp.insert(temp.begin() + k, i);
		outputs.push_back(temp);
	}

}

void My_Extended_Poll::fill_point(const NOMAD::Point & x, vector<NOMAD::Point> *extended) {

	vector<int> in_list;
	vector< vector<int> > outputs{};
	NOMAD::Signature *point_sig;

	for (size_t k = 2; k < x.size(); k++) {
		in_list.push_back(int(x[k].value()));
	}

	int n_stages = int(in_list.size());
	fill_i(in_list, n_stages, outputs);

	vector<int> in_pt;

	for (size_t n = 0; n < outputs.size(); n++) {
		in_pt = outputs[n];

		int max_stages = int(in_pt.size());

		if (max_stages == 1) {
			point_sig = _s1;
		}
		else if (max_stages == 2) {
			point_sig = _s2;
		}
		else if (max_stages == 3) {
			point_sig = _s3;
		}
		else if (max_stages == 4) {
			point_sig = _s4;
		}
		else if (max_stages == 5) {
			point_sig = _s5;
		}
		else if (max_stages == 6) {
			point_sig = _s6;
		}

		NOMAD::Point y(max_stages + 2);
		y[0] = max_stages;
		y[1] = x[1];

		for (size_t k = 2; k < max_stages + 2; k++) {
			y[k] = in_pt[k - 2];
		}

		add_extended_poll_point(y, *point_sig);
		extended->push_back(y);
	}
}

void My_Extended_Poll::fill_point(const NOMAD::Eval_Point & x, vector<NOMAD::Point> *extended) {

	vector<int> in_list;
	vector< vector<int> > outputs{};
	NOMAD::Signature *point_sig;

	for (size_t k = 2; k < x.size(); k++) {
		in_list.push_back(int(x[k].value()));
	}

	int n_stages = int(in_list.size());
	fill_i(in_list, n_stages, outputs);

	vector<int> in_pt;

	for (size_t n = 0; n < outputs.size(); n++) {
		in_pt = outputs[n];

		int max_stages = int(in_pt.size());

		if (max_stages == 1) {
			point_sig = _s1;
		}
		else if (max_stages == 2) {
			point_sig = _s2;
		}
		else if (max_stages == 3) {
			point_sig = _s3;
		}
		else if (max_stages == 4) {
			point_sig = _s4;
		}
		else if (max_stages == 5) {
			point_sig = _s5;
		}
		else if (max_stages == 6) {
			point_sig = _s6;
		}

		NOMAD::Point y(max_stages + 2);
		y[0] = max_stages;
		y[1] = x[1];

		for (size_t k = 2; k < max_stages + 2; k++) {
			y[k] = in_pt[k - 2];
		}

		add_extended_poll_point(y, *point_sig);
		extended->push_back(y);
	}
}

void My_Extended_Poll::shuffle_padding(const NOMAD::Point & x, vector<NOMAD::Point> *extended) {

	vector<int> in_list;
	int max_stages = int(x[0].value());
	vector< vector<int> > outputs{};
	NOMAD::Signature *point_sig;

	if (max_stages == 1) {
		point_sig = _s1;
	}
	else if (max_stages == 2) {
		point_sig = _s2;
	}
	else if (max_stages == 3) {
		point_sig = _s3;
	}
	else if (max_stages == 4) {
		point_sig = _s4;
	}
	else if (max_stages == 5) {
		point_sig = _s5;
	}
	else if (max_stages == 6) {
		point_sig = _s6;
	}

	for (size_t k = 2; k < x.size(); k++) {
		if (x[k].value() != -1) {
			in_list.push_back(int(x[k].value()));
		}
	}

	int n_deposit = int(in_list.size());
	int n_insert = max_stages - n_deposit;
	insert_i(in_list, n_insert, n_deposit, outputs);

	vector<int> in_pt;

	for (size_t n = 0; n < outputs.size(); n++) {
		in_pt = outputs[n];

		NOMAD::Point y(max_stages + 2);
		y[0] = max_stages;
		y[1] = x[1];

		for (size_t k = 2; k < max_stages + 2; k++) {
			y[k] = in_pt[k - 2];
		}

		add_extended_poll_point(y, *point_sig);
		extended->push_back(y);
	}

}

void My_Extended_Poll::shuffle_padding(const NOMAD::Eval_Point & x, vector<NOMAD::Point> *extended) {

	vector<int> in_list;
	int max_stages = int(x[0].value());
	vector< vector<int> > outputs{};

	NOMAD::Signature *point_sig;

	if (max_stages == 1) {
		point_sig = _s1;
	}
	else if (max_stages == 2) {
		point_sig = _s2;
	}
	else if (max_stages == 3) {
		point_sig = _s3;
	}
	else if (max_stages == 4) {
		point_sig = _s4;
	}
	else if (max_stages == 5) {
		point_sig = _s5;
	}
	else if (max_stages == 6) {
		point_sig = _s6;
	}

	for (size_t k = 2; k < x.size(); k++) {
		if (x[k].value() != -1) {
			in_list.push_back(int(x[k].value()));
		}
	}

	int n_deposit = int(in_list.size());
	int n_insert = max_stages - n_deposit;
	insert_i(in_list, n_insert, n_deposit, outputs);

	vector<int> in_pt;

	for (size_t n = 0; n < outputs.size(); n++) {
		in_pt = outputs[n];

		NOMAD::Point y(max_stages + 2);
		y[0] = max_stages;
		y[1] = x[1];

		for (size_t k = 2; k < max_stages + 2; k++) {
			y[k] = in_pt[k - 2];
		}

		add_extended_poll_point(y, *point_sig);
		extended->push_back(y);
	}
}

void My_Extended_Poll::construct_extended_points(const NOMAD::Eval_Point & x) {

	// number of deposits:
	int n = static_cast<int> (x[0].value());
	vector<NOMAD::Point> extended;

	// current type of concept:
	int c = static_cast<int> (x[1].value());

	// current type of deposit
	size_t n_var = x.size(); // Accessing last element 
	int cur_type = static_cast<int> (x[n_var - 1].value());

	// list of concepts
	vector<int> concepts = { 0,1 };
	vector<int> deposits_c1 = { 0,1,2,3 };
	vector<int> deposits_c0 = { 0,1,2 };
	vector<int> deposits;
	// this vector contains the types of the other deposits:
	vector<int> other_types, other_concepts, other_types_change, other_types_add;

	// types of deposits available to each concept
	switch (c) {
	case 0:
		deposits = deposits_c0;
		break;

	case 1:
		deposits = deposits_c1;
		break;
	}

	// remove existing deposits from available choices:
	for (size_t k = 0; k < (x.size() - 2); ++k) {
		deposits.erase(remove(deposits.begin(), deposits.end(), x[k + 2]), deposits.end());
	}

	// -1 not allowed for first selection
	if (n > 1) {
		deposits.push_back(-1); // add the "do nothing" option back
	}

	other_concepts.push_back(0);
	other_types = deposits;

	other_types_change = other_types; // do not change into same deposit type
	other_types_change.erase(
		remove(other_types_change.begin(), other_types_change.end(), cur_type),
		other_types_change.end());

	other_types_add = other_types;
	other_types_add.push_back(-1); // -1 allowed only if adding a stiffener

	// 1 deposit:
	// --------
	if (n == 1) {

		// add 1 deposit (1 or 3 neighbors):
		for (size_t k = 0; k < other_types_add.size(); ++k) {
			NOMAD::Point y(4);

			y[0] = 2;
			y[1] = c;
			y[2] = cur_type;
			y[3] = other_types_add[k];

			add_extended_poll_point(y, *_s2);
			extended.push_back(y);
			fill_point(y, &extended);
		}

		// change the type of the deposit to the other types (1 or 3 neighbors):
		for (size_t k = 0; k < other_types.size(); ++k) {
			NOMAD::Point y = x;
			y[2] = other_types[k];

			add_extended_poll_point(y, *_s1);
			extended.push_back(y);
			fill_point(y, &extended);
		}

		// loop over concepts
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
				}

				// change the type of the deposit to the other types (1 or 3 neighbors):
				for (size_t k = 0; k < other_types.size(); ++k) {
					NOMAD::Point y = x;
					y[1] = other_concepts[j];
					y[2] = other_types[k];

					add_extended_poll_point(y, *_s1);
					extended.push_back(y);
					fill_point(y, &extended);
				}

			}
		}
		fill_point(x, &extended);
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
			fill_point(y, &extended);
		}

		// change the type of one deposit (2 neighbors):

		for (size_t k = 0; k < other_types_change.size(); ++k) {
			NOMAD::Point y = x;
			y[3] = other_types_change[k];

			add_extended_poll_point(y, *_s2);
			extended.push_back(y);
			fill_point(y, &extended);
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
			shuffle_padding(y, &extended);
			fill_point(y, &extended);
		}
		fill_point(x, &extended);
	}

	// 3 deposits:
	// ---------
	else if (n == 3) {

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

		for (size_t k = 0; k < other_types_change.size(); ++k) {
			NOMAD::Point y = x;
			y[4] = other_types_change[k];

			add_extended_poll_point(y, *_s3);
			extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		// add one deposit (1 neighbor):
		for (size_t k = 0; k < other_types.size(); ++k) {
			NOMAD::Point y(6);
			y[0] = 4;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];
			y[4] = cur_type;
			y[5] = other_types[k];

			add_extended_poll_point(y, *_s4);
			extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		shuffle_padding(x, &extended);

	}

	// 4 deposits:
	// ---------
	else if (n == 4) {

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
			shuffle_padding(y, &extended);
		}

		// change the type of one deposit (1 neighbor):
		for (size_t k = 0; k < other_types_change.size(); ++k) {
			NOMAD::Point y = x;
			y[5] = other_types_change[k];

			add_extended_poll_point(y, *_s4);
			extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		// add one deposit (1 neighbor):
		for (size_t k = 0; k < other_types.size(); ++k) {
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
			shuffle_padding(y, &extended);
		}

		shuffle_padding(x, &extended);

	}

	// 5 deposits:
	// ---------
	else if (n == 5) {

		// remove 1 deposit (1 neighbor):
		{
			NOMAD::Point y(6);
			y[0] = 4;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];
			y[4] = x[4];
			y[5] = x[5];

			add_extended_poll_point(y, *_s4);
			extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		// change the type of one deposit (1 neighbor):
		for (size_t k = 0; k < other_types_change.size(); ++k) {
			NOMAD::Point y = x;
			y[6] = other_types_change[k];

			add_extended_poll_point(y, *_s5);
			extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		// add one deposit (1 neighbor):
		for (size_t k = 0; k < other_types.size(); ++k) {
			NOMAD::Point y(8);
			y[0] = 6;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];
			y[4] = x[4];
			y[5] = x[5];
			y[6] = cur_type;
			y[7] = other_types[k];

			add_extended_poll_point(y, *_s6);
			extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		shuffle_padding(x, &extended);

	}

	// 6 deposits:
	// ---------
	else {

		// remove one deposit (1 neighbor):
		NOMAD::Point y(7);
		y[0] = 5;
		y[1] = c;
		y[2] = x[2];
		y[3] = x[3];
		y[4] = x[4];
		y[5] = x[5];
		y[6] = x[6];

		add_extended_poll_point(y, *_s5);
		extended.push_back(y);
		shuffle_padding(y, &extended);
		shuffle_padding(x, &extended);
	}

	//// display extended poll points
	//for (size_t k = 0; k < extended.size(); k++) {

	//	NOMAD::Point p = extended[k];
	//	cout << p << endl;

	//}
	//cout << "==============================" << endl;

}
