#include "autograd.h"
void testEinsor()
{
	// EinsorD tensora(3, 2, true);
	// tensora << 1, 2, 3, 4, 5, 6;
	// EinsorD tensorb(2, 3, true);
	// tensorb << 1, 2, 3, 4, 5, 6;
	// EinsorD tensorc(3, 3, true);
	// tensorc << 1, 1, 1, 1, 1, 1, 1, 1, 1;
	// // tensorc << 2, 2, 2, 2, 2, 2, 2, 2, 2;
	// EinsorD result = tensora * tensorb;
	// result = (result * tensorc).sum();
	// result.backward();
	// // result.backward();
	// cout << tensora;
	// cout << tensorb;
	// cout << tensorc;

	EinsorD tensora(3, 2, true);
	tensora << 1, 2, 3, 4, 5, 6;
	EinsorD tensorb(3, 2, true);
	tensorb << 2, 3, 4, 5, 6, 7;
	auto rab = tensora / tensorb;
	auto rabd = rab.sum();
	rabd.backward();
	cout << tensora;
	cout << tensorb;
}