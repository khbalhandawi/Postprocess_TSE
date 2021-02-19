#pragma once

#include "nomad.hpp"

#ifndef MY_EXTENDED_POLL_BIOBJ_H
#define MY_EXTENDED_POLL_BIOBJ_H

/*--------------------------------------------------*/
/*  user class to define categorical neighborhoods  */
/*--------------------------------------------------*/
class My_Extended_Poll : public NOMAD::Extended_Poll
{

private:

	// signatures for 1, 2, 3, 4, or 5 assets:
	NOMAD::Signature * _s1, *_s2, *_s3, *_s4, *_s5;

public:

	/*----------------------------------------*/
	/*               Constructor              */
	/*----------------------------------------*/
	My_Extended_Poll(NOMAD::Parameters &);

	/*----------------------------------------*/
	/*               Destructor               */
	/*----------------------------------------*/
	virtual ~My_Extended_Poll(void) { delete _s1; delete _s2; delete _s3; delete _s4; delete _s5;}

	// construct the extended poll points:
	virtual void construct_extended_points(const NOMAD::Eval_Point &);

};

#endif // MY_EXTENDED_POLL_BIOBJ_H