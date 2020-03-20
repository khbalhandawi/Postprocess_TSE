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

// This function reads a csv delimlited file
vector< vector<double> >  read_csv_file(string filename)
{
	ifstream file(filename); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
	string line, value;
	vector<double> line_vec;
	vector<string> title_vec;
	vector<double> concept, i1, i2, i3, i4, n_f_th, weight, resiliance_th;
	vector< vector<double> > array_grid;

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
			concept.push_back((line_vec[1]));
			i1.push_back(line_vec[2]);
			i2.push_back(line_vec[3]);
			i3.push_back(line_vec[4]);
			i4.push_back(line_vec[5]);
			n_f_th.push_back(line_vec[32]);
			weight.push_back(line_vec[34]);
			resiliance_th.push_back(line_vec[45]);
		}

		i++;
	}
	array_grid = { concept, i1, i2, i3, i4, n_f_th, weight, resiliance_th };
	return array_grid;
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
	My_Evaluator ( const NOMAD::Parameters & p ) :
    Evaluator ( p ) {}
	
	~My_Evaluator ( void ) {}
	
	bool eval_x (NOMAD::Eval_Point   & x          ,
				 const NOMAD::Double & h_max      ,
				 bool         & count_eval   ) const;

};


/*--------------------------------------------------*/
/*  user class to define categorical neighborhoods  */
/*--------------------------------------------------*/
class My_Extended_Poll : public NOMAD::Extended_Poll
{
	
private:
	
	// signatures for 1, 2, 3 or 4 assets:
	NOMAD::Signature * _s1;

public:
	
	// constructor:
	My_Extended_Poll (NOMAD::Parameters & );
	
	// destructor:
	virtual ~My_Extended_Poll ( void ) { delete _s1; }
	
	// construct the extended poll points:
	virtual void construct_extended_points ( const NOMAD::Eval_Point & );
	
};

/*------------------------------------------*/
/*            NOMAD main function           */
/*------------------------------------------*/
int main ( int argc , char ** argv )
{
	
	// NOMAD initializations:
	NOMAD::begin ( argc , argv );
	
	// display:
	NOMAD::Display out ( cout );
	out.precision (NOMAD::DISPLAY_PRECISION_STD );

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
	
	try
	{
		
		// parameters creation:
		NOMAD::Parameters p ( out );
		
		vector<NOMAD::bb_output_type> bbot(2);
		bbot[0] = NOMAD::EB;  // resiliance constraint
		bbot[1] = NOMAD::OBJ; // objective

		// initial point
		NOMAD::Point x0(8, 2);
		x0[0] =  1;  // 1 deposit
		x0[1] =  0;  // wave concept
		x0[2] =  1;  // deposit type 1
		x0[3] = -1;  // deposit type 1
		x0[4] = -1;  // deposit type 1
		x0[5] = -1;  // deposit type 1
		x0[6] = -1;  // deposit type 1
		x0[7] = -1;  // deposit type 1
		p.set_X0(x0);

		NOMAD::Point lb(8);
		NOMAD::Point ub(8);
		// Categorical variables don't need bounds
		lb[1] =  0; ub[1] = 1;
        lb[2] = -1; ub[2] = 3;
		lb[3] = -1; ub[3] = 3;
		lb[4] = -1; ub[4] = 3;
		lb[5] = -1; ub[5] = 3;
		lb[6] = -1; ub[6] = 3;
		lb[7] = -1; ub[7] = 3;

        //p.set_DISPLAY_DEGREE ( NOMAD::FULL_DISPLAY );

		p.set_DIMENSION (8);

		p.set_BB_OUTPUT_TYPE ( bbot );
		
		// categorical variables:
		p.set_BB_INPUT_TYPE ( 0 , NOMAD::CATEGORICAL);
		p.set_BB_INPUT_TYPE ( 1 , NOMAD::INTEGER);
		p.set_BB_INPUT_TYPE ( 2 , NOMAD::INTEGER);
		p.set_BB_INPUT_TYPE ( 3 , NOMAD::INTEGER);
		p.set_BB_INPUT_TYPE ( 4 , NOMAD::INTEGER);
		p.set_BB_INPUT_TYPE ( 5 , NOMAD::INTEGER);
		p.set_BB_INPUT_TYPE ( 6 , NOMAD::INTEGER);
		p.set_BB_INPUT_TYPE ( 7 , NOMAD::INTEGER);

		p.set_DISABLE_MODELS();
		p.set_INITIAL_MESH_SIZE(2.0);

		p.set_LOWER_BOUND ( lb );
		p.set_UPPER_BOUND ( ub );
		
		p.set_DISPLAY_STATS ( "bbe ( sol ) obj" );
		p.set_MAX_BB_EVAL ( 200 );

		// extended poll trigger:
		p.set_EXTENDED_POLL_TRIGGER ( 1 , true );
		
		// parameters validation:
		p.check();

		//// custom point to debug ev and ep:
		//NOMAD::Eval_Point xt(3, 2);
		//xt[0] = 1;
		//xt[1] = 0;
		//xt[2] = 1;

		// custom point to debug ev and ep:
		NOMAD::Eval_Point xt(8, 2);
		xt[0] =  3;
		xt[1] =  1;
		xt[2] =  1;
		xt[3] =  -1;
		xt[4] =  2;
		xt[5] =  -1;
		xt[6] =  -1;
		xt[7] =  -1;

		NOMAD::Double hmax = 2;
		bool count = false;

		// custom evaluator creation:
		My_Evaluator ev (p);

		ev.eval_x(xt, hmax, count);
		xt.display_eval(cout); // Debug evaluator

		// extended poll:
		My_Extended_Poll ep ( p );
		ep.construct_extended_points(xt); // debug extended poll

		// clear bb eval log file:
		ofstream file("mads_bb_calls.log", ofstream::out);
		file.close();

		// algorithm creation and execution:
		NOMAD::Mads mads ( p , &ev , &ep , NULL , NULL );
		mads.run();

		// Write optimal point to file
		const NOMAD::Eval_Point *x_opt;
		x_opt = mads.get_best_feasible();

		ofstream output_file("mads_x_opt.log", ofstream::out);
		x_opt->display(output_file);
		output_file.close();

	}
	catch ( exception & e ) {
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
	bool         & count_eval) const
{
	NOMAD::Double f, g1; // objective function
	//--------------------------------------------------------------//
	// Read from data file
	
	count_eval = false;

	string input_file = "varout_opt_log.log";
	vector< vector<double> > input_array = read_csv_file(input_file); // Make sure there are no empty lines !!

	vector<double> concept, i1, i2, i3, i4, n_f_th, weight, resiliance_th;
	concept = input_array[0];
	i1 = input_array[1];
	i2 = input_array[2];
	i3 = input_array[3];
	i4 = input_array[4];
	n_f_th = input_array[5];
	weight = input_array[6];
	resiliance_th = input_array[7];
	// number of deposits:
	int n = static_cast<int> (x[0].value());
	int value;

	vector<int> input_deposits; // extract different deposit types
	// remove -1 deposits from input:
	for (size_t k = 0; k < (x.size() - 1); ++k) {
		if (x[k + 1].value() != -1) {
			input_deposits.push_back(x[k + 1].value()); // get input vector
		}
	}

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

	vector<int> output_log; // map input variable to output vector
	for (size_t k = 0; k < (x.size() - 1); ++k) {
		output_log.push_back(x[k + 1].value()); // populate output vector
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
			f = weight[i];
			g1 = 0.99 - resiliance_th[i];
			x.set_bb_output(0, g1);
			x.set_bb_output(1, f);
			count_eval = true;
            found = true;

			ofstream file("mads_bb_calls.log", ofstream::app);
			writeTofile(output_log, & file);

			//cout << "objective: " << f << endl;
			// terminate the loop
			break;
		}
	}
    if ( !found )
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
	_s1(NULL)
{
	// signature for 6 stages:
	// ----------------------
	vector<NOMAD::bb_input_type> bbit_1(8);
	bbit_1[0] = NOMAD::CATEGORICAL;
	bbit_1[1] = NOMAD::INTEGER;
	bbit_1[2] = NOMAD::INTEGER;
	bbit_1[3] = NOMAD::INTEGER;
	bbit_1[4] = NOMAD::INTEGER;
	bbit_1[5] = NOMAD::INTEGER;
	bbit_1[6] = NOMAD::INTEGER;
	bbit_1[7] = NOMAD::INTEGER;

	const NOMAD::Point & d0_1 = p.get_initial_poll_size();
	const NOMAD::Point & lb_1 = p.get_lb();
	const NOMAD::Point & ub_1 = p.get_ub();

	_s1 = new NOMAD::Signature(8,
		bbit_1,
		d0_1,
		lb_1,
		ub_1,
		p.get_direction_types(),
		p.get_sec_poll_dir_types(),
		p.get_int_poll_dir_types(),
		_p.out());

}

/*--------------------------------------*/
/*  construct the extended poll points  */
/*      (categorical neighborhoods)     */
/*--------------------------------------*/
void My_Extended_Poll::construct_extended_points(const NOMAD::Eval_Point & x) {

	// number of deposits:
	int n = static_cast<int> (x[0].value());
	vector<NOMAD::Point> extended;

	// current type of concept:
	int c = static_cast<int> (x[1].value());

	// current type of deposit
	int n_var = x[0].value(); // Accessing last element 
	int cur_type = static_cast<int> (x[n_var + 1].value());

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

		// remove existing deposits from available choices:
		for (size_t k = 0; k < (x.size() - 2); ++k) {
			deposits.erase(remove(deposits.begin(), deposits.end(), x[k + 2]), deposits.end());
		}

		// -1 not allowed for first selection
		if (n > 1) {
			deposits.push_back(-1); // add the "do nothing" option back
		}

		other_concepts.push_back(1);
		other_types = deposits;

		other_types_change = other_types; // do not change into same deposit type
		other_types_change.erase(
			remove(other_types_change.begin(), other_types_change.end(), cur_type),
			other_types_change.end());

		other_types_add = other_types;
		other_types_add.push_back(-1); // -1 allowed only if adding a stiffener

		break;

	case 1:

		deposits = deposits_c1;

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

		break;
	}

	// 1 deposit:
	// --------
	if (n == 1) {

		// add 1 deposit (1 or 3 neighbors):
		for (size_t k = 0; k < other_types_add.size(); ++k) {
			NOMAD::Point y(8);

			int n_deposits = 2;

			y[0] = n_deposits;
			y[1] = c;
			y[2] = cur_type;
			y[3] = other_types_add[k];
			// set remainder to -1:
			int n_append;
			for (int k = 0; k < (y.size() - (n_deposits + 2)); ++k) {
				n_append = k + (n_deposits + 2);
				y[n_append] = -1;
			}

			add_extended_poll_point(y, *_s1);
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
				for (size_t k = 0; k < other_types.size(); ++k)
				{
					NOMAD::Point y = x;
					y[1] = other_concepts[j];
					y[2] = other_types[k];

					add_extended_poll_point(y, *_s1);
					extended.push_back(y);
				}

			}
		}

	}

	// 2 deposits:
	// --------
	else if (n == 2) {

		// remove 1 deposit (1 neighbor):
		{
			NOMAD::Point y(8);
			int n_deposits = 1;

			y[0] = n_deposits;
			y[1] = c;
			y[2] = x[2];
			// set remainder to -1:
			int n_append;
			for (int k = 0; k < (y.size() - (n_deposits + 2)); ++k) {
				n_append = k + (n_deposits + 2);
				y[n_append] = -1;
			}

			add_extended_poll_point(y, *_s1);
			extended.push_back(y);
		}

		// change the type of one deposit (2 neighbors):

		for (size_t k = 0; k < other_types_change.size(); ++k)
		{
			NOMAD::Point y = x;
			y[3] = other_types_change[k];

			add_extended_poll_point(y, *_s1);
			extended.push_back(y);
		}

		// add one deposit (2 neighbors):
		for (size_t k = 0; k < other_types.size(); ++k)
		{
			NOMAD::Point y(8);
			int n_deposits = 3;

			y[0] = n_deposits;
			y[1] = c;
			y[2] = x[2];
			y[3] = cur_type;
			y[4] = other_types[k];
			// set remainder to -1:
			int n_append;
			for (int k = 0; k < (y.size() - (n_deposits + 2)); ++k) {
				n_append = k + (n_deposits + 2);
				y[n_append] = -1;
			}

			add_extended_poll_point(y, *_s1);
			extended.push_back(y);
		}

	}

	// 3 deposits:
	// ---------
	else if (n == 3)
	{

		// remove 1 deposit (1 neighbor):
		{
			NOMAD::Point y(8);
			int n_deposits = 2;

			y[0] = 2;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];
			// set remainder to -1:
			int n_append;
			for (int k = 0; k < (y.size() - (n_deposits + 2)); ++k) {
				n_append = k + (n_deposits + 2);
				y[n_append] = -1;
			}

			add_extended_poll_point(y, *_s1);
			extended.push_back(y);
		}

		// change the type of one deposit (1 neighbor):

		for (size_t k = 0; k < other_types_change.size(); ++k)
		{
			NOMAD::Point y = x;
			y[4] = other_types_change[k];

			add_extended_poll_point(y, *_s1);
			extended.push_back(y);
		}

		// add one deposit (1 neighbor):
		for (size_t k = 0; k < other_types.size(); ++k)
		{
			NOMAD::Point y(8);
			int n_deposits = 4;
			y[0] = n_deposits;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];
			y[4] = cur_type;
			y[5] = other_types[k];
			// set remainder to -1:
			int n_append;
			for (int k = 0; k < (y.size() - (n_deposits + 2)); ++k) {
				n_append = k + (n_deposits + 2);
				y[n_append] = -1;
			}

			add_extended_poll_point(y, *_s1);
			extended.push_back(y);
		}

	}

	// 4 deposits:
	// ---------
	else if (n == 4)
	{

	// remove 1 deposit (1 neighbor):
	{
		NOMAD::Point y(8);
		int n_deposits = 3;

		y[0] = n_deposits;
		y[1] = c;
		y[2] = x[2];
		y[3] = x[3];
		y[4] = x[4];
		// set remainder to -1:
		int n_append;
		for (int k = 0; k < (y.size() - (n_deposits + 2)); ++k) {
			n_append = k + (n_deposits + 2);
			y[n_append] = -1;
		}

		add_extended_poll_point(y, *_s1);
		extended.push_back(y);
	}

	// change the type of one deposit (1 neighbor):
	for (size_t k = 0; k < other_types_change.size(); ++k)
	{
		NOMAD::Point y = x;
		y[5] = other_types_change[k];

		add_extended_poll_point(y, *_s1);
		extended.push_back(y);
	}

	// add one deposit (1 neighbor):
	for (size_t k = 0; k < other_types.size(); ++k)
	{
		NOMAD::Point y(8);
		int n_deposits = 5;

		y[0] = n_deposits;
		y[1] = c;
		y[2] = x[2];
		y[3] = x[3];
		y[4] = x[4];
		y[5] = cur_type;
		y[6] = other_types[k];
		// set remainder to -1:
		int n_append;
		for (int k = 0; k < (y.size() - (n_deposits + 2)); ++k) {
			n_append = k + (n_deposits + 2);
			y[n_append] = -1;
		}

		add_extended_poll_point(y, *_s1);
		extended.push_back(y);
	}

	}

	// 5 deposits:
	// ---------
	else if (n == 5)
	{

	// remove 1 deposit (1 neighbor):
	{
		NOMAD::Point y(8);
		int n_deposits = 4;

		y[0] = n_deposits;
		y[1] = c;
		y[2] = x[2];
		y[3] = x[3];
		y[4] = x[4];
		y[5] = x[5];
		// set remainder to -1:
		int n_append;
		for (int k = 0; k < (y.size() - (n_deposits + 2)); ++k) {
			n_append = k + (n_deposits + 2);
			y[n_append] = -1;
		}

		add_extended_poll_point(y, *_s1);
		extended.push_back(y);
	}

	// change the type of one deposit (1 neighbor):
	for (size_t k = 0; k < other_types_change.size(); ++k)
	{
		NOMAD::Point y = x;
		y[6] = other_types_change[k];

		add_extended_poll_point(y, *_s1);
		extended.push_back(y);
	}

	// add one deposit (1 neighbor):
	for (size_t k = 0; k < other_types.size(); ++k)
	{
		NOMAD::Point y(8);
		int n_deposits = 6;

		y[0] = 6;
		y[1] = c;
		y[2] = x[2];
		y[3] = x[3];
		y[4] = x[4];
		y[5] = x[5];
		y[6] = cur_type;
		y[7] = other_types[k];
		// set remainder to -1:
		int n_append;
		for (int k = 0; k < (y.size() - (n_deposits + 2)); ++k) {
			n_append = k + (n_deposits + 2);
			y[n_append] = -1;
		}

		add_extended_poll_point(y, *_s1);
		extended.push_back(y);
	}

	}

	// 6 deposits:
	// ---------
	else {

		// remove one deposit (1 neighbor):
		NOMAD::Point y(8);
		int n_deposits = 5;

		y[0] = n_deposits;
		y[1] = c;
		y[2] = x[2];
		y[3] = x[3];
		y[4] = x[4];
		y[5] = x[5];
		y[6] = x[6];
		// set remainder to -1:
		int n_append;
		for (int k = 0; k < (y.size() - (n_deposits + 2)); ++k) {
			n_append = k + (n_deposits + 2);
			y[n_append] = -1;
		}

		add_extended_poll_point(y, *_s1);
		extended.push_back(y);
	}

	// display extended poll points
	for (size_t k = 0; k < extended.size(); k++) {

		NOMAD::Point p = extended[k];
		cout << p << endl;

	}

}
