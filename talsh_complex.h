/** ExaTensor::TAL-SH: Complex arithmetic header.
REVISION: 2019/04/17

Copyright (C) 2014-2019 Dmitry I. Lyakh (Liakh)
Copyright (C) 2014-2019 Oak Ridge National Laboratory (UT-Battelle)

This file is part of ExaTensor.

ExaTensor is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ExaTensor is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with ExaTensor. If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------
**/

#ifndef TALSH_COMPLEX_H_
#define TALSH_COMPLEX_H_

#include <math.h>

#ifdef __cplusplus
#include <complex>
#endif

#ifndef NO_GPU
#include <cuComplex.h>
#endif

//DECLARATIONS:
// Complex number:
#ifndef NO_GPU
typedef cuFloatComplex talshComplex4;
typedef cuDoubleComplex talshComplex8;
#else
#ifdef __cplusplus
typedef std::complex<float> talshComplex4;
typedef std::complex<double> talshComplex8;
#else
typedef struct{float real; float imag;} talshComplex4;
typedef struct{double real; double imag;} talshComplex8;
#endif
#endif /*NO_GPU*/

/* TAL-SH complex arithmetic headers:
inline talshComplex4 talshComplex4Set(float real, float imag);
inline talshComplex8 talshComplex8Set(double real, double imag);
inline float talshComplex4Real(talshComplex4 cmplx);
inline double talshComplex8Real(talshComplex8 cmplx);
inline float talshComplex4Imag(talshComplex4 cmplx);
inline double talshComplex8Imag(talshComplex8 cmplx);
inline talshComplex4 talshComplex4Conjg(talshComplex4 cmplx);
inline talshComplex8 talshComplex8Conjg(talshComplex8 cmplx);
inline float talshComplex4Abs(talshComplex4 cmplx);
inline double talshComplex8Abs(talshComplex8 cmplx);
inline float talshComplex4Asq(talshComplex4 cmplx);
inline double talshComplex8Asq(talshComplex8 cmplx);
inline talshComplex4 talshComplex4Add(talshComplex4 x, talshComplex4 y);
inline talshComplex8 talshComplex8Add(talshComplex8 x, talshComplex8 y);
inline void talshComplex4AddEq(talshComplex4 * x, talshComplex4 y);
inline void talshComplex8AddEq(talshComplex8 * x, talshComplex8 y);
inline talshComplex4 talshComplex4Sub(talshComplex4 x, talshComplex4 y);
inline talshComplex8 talshComplex8Sub(talshComplex8 x, talshComplex8 y);
inline void talshComplex4SubEq(talshComplex4 * x, talshComplex4 y);
inline void talshComplex8SubEq(talshComplex8 * x, talshComplex8 y);
inline talshComplex4 talshComplex4Mul(talshComplex4 x, talshComplex4 y);
inline talshComplex8 talshComplex8Mul(talshComplex8 x, talshComplex8 y);
inline void talshComplex4MulEq(talshComplex4 * x, talshComplex4 y);
inline void talshComplex8MulEq(talshComplex8 * x, talshComplex8 y);
inline talshComplex4 talshComplex4Div(talshComplex4 x, talshComplex4 y);
inline talshComplex8 talshComplex8Div(talshComplex8 x, talshComplex8 y);
inline void talshComplex4DivEq(talshComplex4 * x, talshComplex4 y);
inline void talshComplex8DivEq(talshComplex8 * x, talshComplex8 y);
*/


//DEFINITIONS:
//Construct a complex number:
#ifndef NO_GPU
__host__ __device__ __forceinline__ talshComplex4 talshComplex4Set(float real, float imag)
{
 return make_cuFloatComplex(real,imag);
}
__host__ __device__ __forceinline__ talshComplex8 talshComplex8Set(double real, double imag)
{
 return make_cuDoubleComplex(real,imag);
}
#else
#ifdef __cplusplus
inline talshComplex4 talshComplex4Set(float real, float imag)
{
 return talshComplex4(real,imag);
}
inline talshComplex8 talshComplex8Set(double real, double imag)
{
 return talshComplex8(real,imag);
}
#else
talshComplex4 talshComplex4Set(float real, float imag)
{
 talshComplex4 result = {real,imag};
 return result;
}
talshComplex8 talshComplex8Set(double real, double imag)
{
 talshComplex8 result = {real,imag};
 return result;
}
#endif
#endif

//Get the real component of a complex number:
#ifndef NO_GPU
__host__ __device__ __forceinline__ float talshComplex4Real(talshComplex4 cmplx)
{
 return cuCrealf(cmplx);
}
__host__ __device__ __forceinline__ double talshComplex8Real(talshComplex8 cmplx)
{
 return cuCreal(cmplx);
}
#else
#ifdef __cplusplus
inline float talshComplex4Real(talshComplex4 cmplx)
{
 return cmplx.real();
}
inline double talshComplex8Real(talshComplex8 cmplx)
{
 return cmplx.real();
}
#else
float talshComplex4Real(talshComplex4 cmplx)
{
 return cmplx.real;
}
double talshComplex8Real(talshComplex8 cmplx)
{
 return cmplx.real;
}
#endif
#endif

//Get the real component of a complex number:
#ifndef NO_GPU
__host__ __device__ __forceinline__ float talshComplexReal(talshComplex4 cmplx)
{
 return cuCrealf(cmplx);
}
__host__ __device__ __forceinline__ double talshComplexReal(talshComplex8 cmplx)
{
 return cuCreal(cmplx);
}
#else
#ifdef __cplusplus
inline float talshComplexReal(talshComplex4 cmplx)
{
 return cmplx.real();
}
inline double talshComplexReal(talshComplex8 cmplx)
{
 return cmplx.real();
}
#else
float talshComplexReal(talshComplex4 cmplx)
{
 return cmplx.real;
}
double talshComplexReal(talshComplex8 cmplx)
{
 return cmplx.real;
}
#endif
#endif

//Get the imaginary component of a complex number:
#ifndef NO_GPU
__host__ __device__ __forceinline__ float talshComplex4Imag(talshComplex4 cmplx)
{
 return cuCimagf(cmplx);
}
__host__ __device__ __forceinline__ double talshComplex8Imag(talshComplex8 cmplx)
{
 return cuCimag(cmplx);
}
#else
#ifdef __cplusplus
inline float talshComplex4Imag(talshComplex4 cmplx)
{
 return cmplx.imag();
}
inline double talshComplex8Imag(talshComplex8 cmplx)
{
 return cmplx.imag();
}
#else
float talshComplex4Imag(talshComplex4 cmplx)
{
 return cmplx.imag;
}
double talshComplex8Imag(talshComplex8 cmplx)
{
 return cmplx.imag;
}
#endif
#endif

//Get the imaginary component of a complex number:
#ifndef NO_GPU
__host__ __device__ __forceinline__ float talshComplexImag(talshComplex4 cmplx)
{
 return cuCimagf(cmplx);
}
__host__ __device__ __forceinline__ double talshComplexImag(talshComplex8 cmplx)
{
 return cuCimag(cmplx);
}
#else
#ifdef __cplusplus
inline float talshComplexImag(talshComplex4 cmplx)
{
 return cmplx.imag();
}
inline double talshComplexImag(talshComplex8 cmplx)
{
 return cmplx.imag();
}
#else
float talshComplexImag(talshComplex4 cmplx)
{
 return cmplx.imag;
}
double talshComplexImag(talshComplex8 cmplx)
{
 return cmplx.imag;
}
#endif
#endif

//Get the complex conjugate:
#ifndef NO_GPU
__host__ __device__ __forceinline__ talshComplex4 talshComplex4Conjg(talshComplex4 cmplx)
{
 return cuConjf(cmplx);
}
__host__ __device__ __forceinline__ talshComplex8 talshComplex8Conjg(talshComplex8 cmplx)
{
 return cuConj(cmplx);
}
#else
#ifdef __cplusplus
inline talshComplex4 talshComplex4Conjg(talshComplex4 cmplx)
{
 return std::conj(cmplx);
}
inline talshComplex8 talshComplex8Conjg(talshComplex8 cmplx)
{
 return std::conj(cmplx);
}
#else
talshComplex4 talshComplex4Conjg(talshComplex4 cmplx)
{
 talshComplex4 result = {cmplx.real,-cmplx.imag};
 return result;
}
talshComplex8 talshComplex8Conjg(talshComplex8 cmplx)
{
 talshComplex8 result = {cmplx.real,-cmplx.imag};
 return result;
}
#endif
#endif

//Get the absolute magnitude of a complex number:
#ifndef NO_GPU
__host__ __device__ __forceinline__ float talshComplex4Abs(talshComplex4 cmplx)
{
 return cuCabsf(cmplx);
}
__host__ __device__ __forceinline__ double talshComplex8Abs(talshComplex8 cmplx)
{
 return cuCabs(cmplx);
}
#else
#ifdef __cplusplus
inline float talshComplex4Abs(talshComplex4 cmplx)
{
 return std::abs(cmplx);
}
inline double talshComplex8Abs(talshComplex8 cmplx)
{
 return std::abs(cmplx);
}
#else
float talshComplex4Abs(talshComplex4 cmplx)
{
 return (float)sqrt((double)((cmplx.real)*(cmplx.real)) + (double)((cmplx.imag)*(cmplx.imag)));
}
double talshComplex8Abs(talshComplex8 cmplx)
{
 return sqrt(((cmplx.real)*(cmplx.real)) + ((cmplx.imag)*(cmplx.imag)));
}
#endif
#endif

//Get the squared magnitude of a complex number:
#ifndef NO_GPU
__host__ __device__ __forceinline__ float talshComplex4Asq(talshComplex4 cmplx)
{
 float rl = cuCrealf(cmplx); float im = cuCimagf(cmplx);
 return (rl*rl + im*im);
}
__host__ __device__ __forceinline__ double talshComplex8Asq(talshComplex8 cmplx)
{
 double rl = cuCreal(cmplx); double im = cuCimag(cmplx);
 return (rl*rl + im*im);
}
#else
#ifdef __cplusplus
inline float talshComplex4Asq(talshComplex4 cmplx)
{
 float rl = cmplx.real(); float im = cmplx.imag();
 return (rl*rl + im*im);
}
inline double talshComplex8Asq(talshComplex8 cmplx)
{
 double rl = cmplx.real(); double im = cmplx.imag();
 return (rl*rl + im*im);
}
#else
float talshComplex4Asq(talshComplex4 cmplx)
{
 return ((cmplx.real)*(cmplx.real) + (cmplx.imag)*(cmplx.imag));
}
double talshComplex8Asq(talshComplex8 cmplx)
{
 return ((cmplx.real)*(cmplx.real) + (cmplx.imag)*(cmplx.imag));
}
#endif
#endif

//Add two complex numbers:
#ifndef NO_GPU
__host__ __device__ __forceinline__ talshComplex4 talshComplex4Add(talshComplex4 x, talshComplex4 y)
{
 return cuCaddf(x,y);
}
__host__ __device__ __forceinline__ talshComplex8 talshComplex8Add(talshComplex8 x, talshComplex8 y)
{
 return cuCadd(x,y);
}
#else
#ifdef __cplusplus
inline talshComplex4 talshComplex4Add(talshComplex4 x, talshComplex4 y)
{
 return x+y;
}
inline talshComplex8 talshComplex8Add(talshComplex8 x, talshComplex8 y)
{
 return x+y;
}
#else
talshComplex4 talshComplex4Add(talshComplex4 x, talshComplex4 y)
{
 return talshComplex4Set(x.real+y.real,x.imag+y.imag);
}
talshComplex8 talshComplex8Add(talshComplex8 x, talshComplex8 y)
{
 return talshComplex8Set(x.real+y.real,x.imag+y.imag);
}
#endif
#endif

//Add two complex numbers in-place:
#ifndef NO_GPU
__host__ __device__ __forceinline__ void talshComplex4AddEq(talshComplex4 * x, talshComplex4 y)
{
 *x = cuCaddf(*x,y);
 return;
}
__host__ __device__ __forceinline__ void talshComplex8AddEq(talshComplex8 * x, talshComplex8 y)
{
 *x = cuCadd(*x,y);
 return;
}
#else
#ifdef __cplusplus
inline void talshComplex4AddEq(talshComplex4 * x, talshComplex4 y)
{
 *x = *x + y;
 return;
}
inline void talshComplex8AddEq(talshComplex8 * x, talshComplex8 y)
{
 *x = *x + y;
 return;
}
#else
void talshComplex4AddEq(talshComplex4 * x, talshComplex4 y)
{
 *x = talshComplex4Set(x->real+y.real,x->imag+y.imag);
 return;
}
void talshComplex8AddEq(talshComplex8 * x, talshComplex8 y)
{
 *x = talshComplex8Set(x->real+y.real,x->imag+y.imag);
 return;
}
#endif
#endif

//Subtract two complex numbers:
#ifndef NO_GPU
__host__ __device__ __forceinline__ talshComplex4 talshComplex4Sub(talshComplex4 x, talshComplex4 y)
{
 return cuCsubf(x,y);
}
__host__ __device__ __forceinline__ talshComplex8 talshComplex8Sub(talshComplex8 x, talshComplex8 y)
{
 return cuCsub(x,y);
}
#else
#ifdef __cplusplus
inline talshComplex4 talshComplex4Sub(talshComplex4 x, talshComplex4 y)
{
 return x-y;
}
inline talshComplex8 talshComplex8Sub(talshComplex8 x, talshComplex8 y)
{
 return x-y;
}
#else
talshComplex4 talshComplex4Sub(talshComplex4 x, talshComplex4 y)
{
 return talshComplex4Set(x.real-y.real,x.imag-y.imag);
}
talshComplex8 talshComplex8Sub(talshComplex8 x, talshComplex8 y)
{
 return talshComplex8Set(x.real-y.real,x.imag-y.imag);
}
#endif
#endif

//Multiply two complex numbers:
#ifndef NO_GPU
__host__ __device__ __forceinline__ talshComplex4 talshComplex4Mul(talshComplex4 x, talshComplex4 y)
{
 return cuCmulf(x,y);
}
__host__ __device__ __forceinline__ talshComplex8 talshComplex8Mul(talshComplex8 x, talshComplex8 y)
{
 return cuCmul(x,y);
}
#else
#ifdef __cplusplus
inline talshComplex4 talshComplex4Mul(talshComplex4 x, talshComplex4 y)
{
 return x*y;
}
inline talshComplex8 talshComplex8Mul(talshComplex8 x, talshComplex8 y)
{
 return x*y;
}
#else
talshComplex4 talshComplex4Mul(talshComplex4 x, talshComplex4 y)
{
 float rlx = x.real; float imx = x.imag;
 float rly = y.real; float imy = y.imag;
 return talshComplex4Set(rlx*rly-imx*imy,rlx*imy+imx*rly);
}
talshComplex8 talshComplex8Mul(talshComplex8 x, talshComplex8 y)
{
 double rlx = x.real; double imx = x.imag;
 double rly = y.real; double imy = y.imag;
 return talshComplex8Set(rlx*rly-imx*imy,rlx*imy+imx*rly);
}
#endif
#endif

//Divide two complex numbers:
#ifndef NO_GPU
__host__ __device__ __forceinline__ talshComplex4 talshComplex4Div(talshComplex4 x, talshComplex4 y)
{
 return cuCdivf(x,y);
}
__host__ __device__ __forceinline__ talshComplex8 talshComplex8Div(talshComplex8 x, talshComplex8 y)
{
 return cuCdiv(x,y);
}
#else
#ifdef __cplusplus
inline talshComplex4 talshComplex4Div(talshComplex4 x, talshComplex4 y)
{
 return x/y;
}
inline talshComplex8 talshComplex8Div(talshComplex8 x, talshComplex8 y)
{
 return x/y;
}
#else
talshComplex4 talshComplex4Div(talshComplex4 x, talshComplex4 y)
{
 float rlx = x.real; float imx = x.imag;
 float rly = y.real; float imy = y.imag;
 float dny = 1.0f/(rly*rly + imy*imy);
 return talshComplex4Set((rlx*rly+imx*imy)*dny,(imx*rly-rlx*imy)*dny);
}
talshComplex8 talshComplex8Div(talshComplex8 x, talshComplex8 y)
{
 double rlx = x.real; double imx = x.imag;
 double rly = y.real; double imy = y.imag;
 double dny = 1.0/(rly*rly + imy*imy);
 return talshComplex8Set((rlx*rly+imx*imy)*dny,(imx*rly-rlx*imy)*dny);
}
#endif
#endif

//HELPERS:
template<typename T>
struct ComplexType{
 using Type = T;
 using RealType = void;
 static constexpr bool valid = false;
};

template<>
struct ComplexType<talshComplex4>{
 using Type = talshComplex4;
 using RealType = float;
 static constexpr bool valid = true;
};

template<>
struct ComplexType<talshComplex8>{
 using Type = talshComplex8;
 using RealType = double;
 static constexpr bool valid = true;
};


template<typename T>
struct RealType{
 using Type = T;
 using ComplexType = void;
 static constexpr bool valid = false;
};

template<>
struct RealType<float>{
 using Type = float;
 using ComplexType = talshComplex4;
 static constexpr bool valid = true;
};

template<>
struct RealType<double>{
 using Type = double;
 using ComplexType = talshComplex8;
 static constexpr bool valid = true;
};

#endif /*TALSH_COMPLEX_H_*/
