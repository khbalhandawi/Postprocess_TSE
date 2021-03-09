#include "My_Extended_Poll.h"
#include "user_functions.h"
#include <vector>

/*-----------------------------------------*/
/*  constructor: creates the 6 signatures  */
/*-----------------------------------------*/
My_Extended_Poll::My_Extended_Poll(NOMAD::Parameters & p)
	: Extended_Poll(p),
	_s1(NULL),
	_s2(NULL),
	_s3(NULL),
	_s4(NULL),
	_s5(NULL),
	_s6(NULL) 
{

	NOMAD::bb_input_type n_deposits_sig = NOMAD::CATEGORICAL;
	NOMAD::bb_input_type concept_sig = NOMAD::INTEGER;
	NOMAD::bb_input_type deposit_sig = NOMAD::INTEGER;

	// get signature for initial point
	const NOMAD::Point & d0_0 = p.get_initial_poll_size();
	const NOMAD::Point & lb_0 = p.get_lb();
	const NOMAD::Point & ub_0 = p.get_ub();

	// signature for 1 stage:
	// ----------------------
	std::vector<NOMAD::bb_input_type> bbit_1(3);
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
		std::vector<NOMAD::bb_input_type> bbit_2(4);
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
		std::vector<NOMAD::bb_input_type> bbit_4(6);
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
		std::vector<NOMAD::bb_input_type> bbit_5(7);
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
		std::vector<NOMAD::bb_input_type> bbit_6(8);
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
void My_Extended_Poll::fill_point(const NOMAD::Point & x, vector<NOMAD::Point> *extended)
{

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

void My_Extended_Poll::fill_point(const NOMAD::Eval_Point & x, vector<NOMAD::Point> *extended)
{

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

void My_Extended_Poll::shuffle_padding(const NOMAD::Point & x, vector<NOMAD::Point> *extended)
{

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

void My_Extended_Poll::shuffle_padding(const NOMAD::Eval_Point & x, vector<NOMAD::Point> *extended)
{

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

void My_Extended_Poll::construct_extended_points(const NOMAD::Eval_Point & x)
{

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

	// -1 not allowed for first selection
	if (n > 1) {
		deposits.push_back(-1); // add the "do nothing" option back
	}

	other_types = deposits;

	other_types_change = other_types; // do not change into same deposit type
	other_types_change.erase(
		remove(other_types_change.begin(), other_types_change.end(), cur_type),
		other_types_change.end());

	other_types_add = other_types;
	other_types_add.push_back(-1); // -1 allowed only if adding a stiffener


	// Extract design vector from decision vector (strip the -1 decisions)
	int n_stages = 6;
	vector<int> input_deposits; // extract different deposit types
	input_deposits.push_back(int(x[1].value())); // push back concept type

	for (size_t k = 0; k < (n_stages); ++k) {

		if (k < (x.size() - 2)) {
			input_deposits.push_back(int(x[k + 2].value())); // get input vector
		}

	}

	// remove -1 deposits from input:
	vector<int> lookup_vector;
	for (size_t m = 0; m < input_deposits.size(); ++m) {
		if (input_deposits[m] != -1) {
			lookup_vector.push_back(input_deposits[m]);
		}
	}

	bool check_other_concepts = true;

	//cout << x << endl;

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
					fill_point(y, &extended);
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
						fill_point(y, &extended);
					}
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

			//add_extended_poll_point(y, *_s3);
			//extended.push_back(y);
			shuffle_padding(y, &extended);
			fill_point(y, &extended);
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

			//add_extended_poll_point(y, *_s3);
			//extended.push_back(y);
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

			//add_extended_poll_point(y, *_s4);
			//extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		shuffle_padding(x, &extended);

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
	else if (n == 4) {

		// remove 1 deposit (1 neighbor):
		{
			NOMAD::Point y(5);
			y[0] = 3;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];
			y[4] = x[4];

			//add_extended_poll_point(y, *_s3);
			//extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		// change the type of one deposit (1 neighbor):
		for (size_t k = 0; k < other_types_change.size(); ++k) {
			NOMAD::Point y = x;
			y[5] = other_types_change[k];

			//add_extended_poll_point(y, *_s4);
			//extended.push_back(y);
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

			//add_extended_poll_point(y, *_s5);
			//extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		shuffle_padding(x, &extended);

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

		// remove 1 deposit (1 neighbor):
		{
			NOMAD::Point y(6);
			y[0] = 4;
			y[1] = c;
			y[2] = x[2];
			y[3] = x[3];
			y[4] = x[4];
			y[5] = x[5];

			//add_extended_poll_point(y, *_s4);
			//extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		// change the type of one deposit (1 neighbor):
		for (size_t k = 0; k < other_types_change.size(); ++k) {
			NOMAD::Point y = x;
			y[6] = other_types_change[k];

			//add_extended_poll_point(y, *_s5);
			//extended.push_back(y);
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

			//add_extended_poll_point(y, *_s6);
			//extended.push_back(y);
			shuffle_padding(y, &extended);
		}

		shuffle_padding(x, &extended);

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

				if (!(find(deposits.begin(), deposits.end(), lookup_vector[a]) != deposits.end())) {
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

		//add_extended_poll_point(y, *_s5);
		//extended.push_back(y);
		shuffle_padding(y, &extended);
		shuffle_padding(x, &extended);

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

				add_extended_poll_point(y, *_s6);
				extended.push_back(y);
			}

		}

	}

	//// display extended poll points
	//for (size_t k = 0; k < extended.size(); k++) {

	//	NOMAD::Point p = extended[k];
	//	cout << p << endl;

	//}
	//cout << "==============================" << endl;

}