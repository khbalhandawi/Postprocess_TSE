
#include "user_functions.h"

#include <sstream>
#include <iterator>
#include <iostream>

/*-----------------------------------------------------------*/
/*               Split a string by delimlitor                */
/*-----------------------------------------------------------*/
std::vector<double> split_string(std::string line)
{
	// This function splits a delimlited string into vector of doubles
	std::string input = line;
	std::istringstream ss(input);
	std::string value;
	double value_db;

	std::vector<double> out_vector;
	while (std::getline(ss, value, ','))
	{
		value_db = std::atof(value.c_str()); // convert to float
		out_vector.push_back(value_db); // Vector of floats
		// cout << token << '\n';
	}

	return out_vector;
}

/*-----------------------------------------------------------*/
/*            Split a title string by delimlitor             */
/*-----------------------------------------------------------*/
std::vector<string> split_string_titles(std::string line)
{
	std::string input = line;
	std::istringstream ss(input);
	std::string value;

	std::vector<string> out_vector;
	while (std::getline(ss, value, ','))
	{
		out_vector.push_back(value.c_str()); // Vector of strings
		// cout << token << '\n';
	}

	return out_vector;
}

/*-----------------------------------------------------------*/
/*             Transpose of a vector of vectors              */
/*-----------------------------------------------------------*/
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double> > data)
{
	// this assumes that all inner vectors have the same size and
	// allocates space for the complete result in advance
	std::vector<std::vector<double>> result(data[0].size(),
		std::vector<double>(data.size()));
	for (std::vector<double>::size_type i = 0; i < data[0].size(); i++)
		for (std::vector<double>::size_type j = 0; j < data.size(); j++) {
			result[i][j] = data[j][i];
		}
	return result;
}

/*-----------------------------------------------------------*/
/*      Read a csv file and output a vector of vectors       */
/*-----------------------------------------------------------*/
std::vector< std::vector<double> >  read_csv_file(std::string filename) 
{
	std::ifstream file(filename); // declare file stream: http://www.cplusplus.com/reference/iostream/ifstream/
	std::string line, value;
	std::vector<double> line_vec;
	std::vector<std::string> title_vec;
	std::vector< std::vector<double> > array_grid, array_grid_T;

	int i = 0;
	while (file.good())
	{
		std::getline(file, line, '\n'); // read a string until next comma: http://www.cplusplus.com/reference/string/getline/


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



/*-----------------------------------------------------------*/
/*				 Print vector of ints to cout			     */
/*-----------------------------------------------------------*/
void print_vector(const std::vector<int> & other_types)
{

	std::ostringstream oss;

	if (!other_types.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		copy(other_types.begin(), other_types.end() - 1,
			std::ostream_iterator<int>(oss, ","));

		// Now add the last element with no delimiter
		oss << other_types.back();
	}

	std::cout << oss.str() << std::endl;

}

/*-----------------------------------------------------------*/
/*			  Print vector of doubles to cout			     */
/*-----------------------------------------------------------*/
void print_vec_double(std::vector<double> const &input)
{
	for (int i = 0; i < input.size(); i++) {
		std::cout << input.at(i) << ' ';
	}
	std::cout << std::endl;
}

/*-----------------------------------------------------------*/
/*		         Write vector of ints to file		         */
/*-----------------------------------------------------------*/
bool writeTofile(std::vector<int> &matrix, std::ofstream *file)
{

	int n_vars = 8;
	std::vector<int> matrix_us; // unstripped matrix

	for (size_t k = 0; k < (n_vars); ++k) {

		if (k < matrix.size()) {
			matrix_us.push_back(matrix[k]);
		}
		else {
			matrix_us.push_back(-1); // append -1 to remaining entries
		}
	}

	// Convert int to ostring stream
	std::ostringstream oss;

	if (!matrix_us.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		copy(matrix_us.begin(), matrix_us.end() - 1,
			std::ostream_iterator<int>(oss, ","));

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

/*-----------------------------------------------------------*/
/*		          Write nomad output to file		         */
/*-----------------------------------------------------------*/
bool writeTofile_output(NOMAD::Point & matrix, std::ofstream *file)
{

	size_t n_outs = matrix.size();
	std::vector<double> matrix_out; // unstripped matrix

	for (size_t k = 0; k < (n_outs); ++k)
	{
		matrix_out.push_back(matrix[k].value());
	}

	// Convert int to ostring stream
	std::ostringstream oss;
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

/*-----------------------------------------------------------*/
/*		     Write output vectors (dboule) to file	         */
/*-----------------------------------------------------------*/
bool writeTofile_vector(std::vector<double> & matrix, std::ofstream *file)
{

	// Convert int to ostring stream
	std::ostringstream oss;
	if (!matrix.empty())
	{
		// Convert all but the last element to avoid a trailing ","
		copy(matrix.begin(), matrix.end() - 1,
			std::ostream_iterator<double>(oss, ","));

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

/*-----------------------------------------------------------*/
/* Generate permutations of a vector by inserting -1 element */
/*-----------------------------------------------------------*/
void insert_i(std::vector<int> in_list, int n_insert, int n_deposit, std::vector<std::vector<int>> &outputs, int depth, int shift)
{

	depth += 1;
	std::vector<int> temp, out;
	int i = -1;
	for (int k = 1 + shift; k < n_insert + n_deposit; k++) {
		temp = in_list;
		if (k > temp.size()) {
			temp.insert(temp.end(), i);
		}
		else {
			temp.insert(temp.begin() + k, i);
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

/*-----------------------------------------------------------*/
/*	   Fill vector with -1's until it reaches a size of 6    */
/*-----------------------------------------------------------*/
void fill_i(std::vector<int> in_list, int n_stages, std::vector< std::vector<int> > &outputs)
{

	std::vector<int> temp, out;
	int i = -1;
	temp = in_list;

	for (int k = n_stages; k < 6; k++) {
		temp.insert(temp.begin() + k, i);
		outputs.push_back(temp);
	}

}
