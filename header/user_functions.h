#pragma once

#include "nomad.hpp"

#include <vector>
#include <string>

#ifndef USER_FUNCTIONS_H
#define USER_FUNCTIONS_H

/*-----------------------------------------------------------*/
/*               Split a string by delimlitor                */
/*-----------------------------------------------------------*/
std::vector<double> split_string(std::string line);

/*-----------------------------------------------------------*/
/*            Split a title string by delimlitor             */
/*-----------------------------------------------------------*/
std::vector<std::string> split_string_titles(std::string line);

/*-----------------------------------------------------------*/
/*             Transpose of a vector of vectors              */
/*-----------------------------------------------------------*/
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double> > data);

/*-----------------------------------------------------------*/
/*      Read a csv file and output a vector of vectors       */
/*-----------------------------------------------------------*/
std::vector< std::vector<double> >  read_csv_file(std::string filename);

/*-----------------------------------------------------------*/
/*				 Print vector of ints to cout			     */
/*-----------------------------------------------------------*/
void print_vector(const std::vector<int> & other_types);

/*-----------------------------------------------------------*/
/*			  Print vector of doubles to cout			     */
/*-----------------------------------------------------------*/
void print_vec_double(std::vector<double> const &input);

/*-----------------------------------------------------------*/
/*		         Write vector of ints to file		         */
/*-----------------------------------------------------------*/
bool writeTofile(std::vector<int> &matrix, std::ofstream *file);

/*-----------------------------------------------------------*/
/*		          Write nomad output to file		         */
/*-----------------------------------------------------------*/
bool writeTofile_output(NOMAD::Point & matrix, std::ofstream *file);

/*-----------------------------------------------------------*/
/*		     Write output vectors (dboule) to file	         */
/*-----------------------------------------------------------*/
bool writeTofile_vector(std::vector<double> & matrix, std::ofstream *file);

/*-----------------------------------------------------------*/
/* Generate permutations of a vector by inserting -1 element */
/*-----------------------------------------------------------*/
void insert_i(std::vector<int> in_list, int n_insert, int n_deposit, std::vector<std::vector<int>> &outputs, int depth = 0, int shift = 0);

/*-----------------------------------------------------------*/
/*	   Fill vector with -1's until it reaches a size of 6    */
/*-----------------------------------------------------------*/
void fill_i(std::vector<int> in_list, int n_stages, std::vector< std::vector<int> > &outputs);

#endif // USER_FUNCTIONS_H