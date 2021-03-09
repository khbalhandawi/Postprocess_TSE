#include "My_Extended_Poll_biobj.h"

#include <vector>
/*-----------------------------------------*/
/*  constructor: creates the 5 signatures  */
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
	std::vector<NOMAD::bb_input_type> bbit_1(3);
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
		std::vector<NOMAD::bb_input_type> bbit_2(4);
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
		std::vector<NOMAD::bb_input_type> bbit_3(5);
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
		std::vector<NOMAD::bb_input_type> bbit_4(6);
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
		std::vector<NOMAD::bb_input_type> bbit_5(7);
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

	// number of deposits:
	int n = static_cast<int> (x[0].value());
	std::vector<NOMAD::Point> extended;

	// current type of concept:
	int c = static_cast<int> (x[1].value());

	// current type of deposit
	size_t n_var = x.size(); // Accessing last element 
	int cur_type = static_cast<int> (x[n_var - 1].value());

	// list of concepts
	std::vector<int> concepts = { 0,1,2 };
	std::vector<int> deposits_c2 = { 0,1,2,3 };
	std::vector<int> deposits_c1 = { 0,1,2,3,4 };
	std::vector<int> deposits_c0 = { 0,1,2 };
	std::vector<int> deposits;
	// this vector contains the types of the other deposits:
	std::vector<int> other_types, other_concepts, other_types_change, other_types_add;


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
	std::vector<int> input_deposits; // extract different deposit types
	input_deposits.push_back(int(x[1].value())); // push back concept type

	for (size_t k = 0; k < (x.size() - 2); ++k) {
		input_deposits.push_back(int(x[k + 2].value())); // get input vector
	}

	// remove -1 deposits from input (in MSSP code):
	std::vector<int> lookup_vector;
	for (size_t m = 0; m < input_deposits.size(); ++m) {
		lookup_vector.push_back(input_deposits[m]);
	}

	bool check_other_concepts = true; // Added other equivalent concepts to extended pool

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

	//for (size_t k = 0; k < extended.size(); k++) {

	//	NOMAD::Point p = extended[k];
	//	std::cout << p << endl::cout;

	//}

}
